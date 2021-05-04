from generator import Generator, load_rgb_img, show_image
from ImageTransformationNN import ImageTransformationNN
from VGG16 import VGG16LossNN
from utils import preprocess, normalize_batch, gram_matrix


import torch, time, argparse, os

from torch.optim import Adam
from torch.nn import MSELoss
from torchvision import transforms
from torch.autograd import set_detect_anomaly, Variable

import numpy as np



class StyleTransferTrainer(object):
    
    def __init__(self, args):
        self.args = args
        self.loss_net = VGG16LossNN()
        self.transformer = ImageTransformationNN()
        
        if self.args.retrain:
            self.transformer.load_state_dict(torch.load(self.args.retrain_model,  map_location=torch.device('cpu')))
            print('Loaded weights successfully ')
        
        self.gen = Generator(self.args.data_dir, self.args.batch)
        print('Loaded images successfully')
        self.style = load_rgb_img(self.args.style)
    
    def train(self):
        
        optimizer = Adam(self.transformer.parameters(), self.args.lr)

        mse_loss = MSELoss()

        style_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.mul(255))
        ])
        
        style = style_transform(self.style)
        style = style.repeat(self.args.batch, 1, 1, 1)
        style = preprocess(style)
        
        loss_net_input = normalize_batch(style)
        
        feature_style = self.loss_net.forward(normalize_batch(style))
        gram_style = [gram_matrix(y) for y in feature_style.values()]
    
        l_total_history, l_feat_history, l_style_history = [], [], []
        l_feat_total_history, l_style_total_history, l_total_total_history = [], [], []     

        for epoch in range(self.args.epochs):

            print('epoch:', epoch)
            self.transformer.train()
            #print('-------------\n', self.transformer, '\n-------------')

            l_feat_total = 0.
            l_style_total = 0.
            count = 0

            # To revise
            for batch_id, (x, _) in enumerate(self.gen):

                # Add the batch size
                n_batch = len(x)
                count += n_batch

                # Adam
                optimizer.zero_grad()

                # Preprocess:
                x = Variable(preprocess(x))

                # Feed x to transformer
                y = self.transformer(x)
                    
                xc = Variable(x.clone())

                # Normalize batch
                y = normalize_batch(y)
                xc = normalize_batch(xc)

                # Features from the VGG16 network
                features_y = self.loss_net(y)
                features_xc = self.loss_net(xc)
                    
                f_xc_c = Variable(features_xc['relu2_2'], requires_grad=False)

                # Update features reconstruction loss
                mse = mse_loss(features_y['relu2_2'], f_xc_c)

                l_feat = self.args.content_weights * mse

                l_style = 0.
                
                for m, k in enumerate(features_y.keys()):
                    gram_s = Variable(gram_style[m].data, requires_grad=False)
                    gram_y = gram_matrix(features_y[k])
                    l_style += self.args.style_weights * mse_loss(gram_y, gram_s[:n_batch, :, :])

                l_total = l_feat + l_style

                l_total.backward()
                optimizer.step()

                l_feat_total += l_feat.item()
                l_style_total += l_style.item()

                if (batch_id + 1) % 10 == 0:

                    # Saving losses per item
                    l_feat_history.append(l_feat.item())
                    l_style_history.append(l_style.item())
                    l_total_history.append(l_total.item())

                    # Total
                    l_feat_total_history.append(l_feat_total / (batch_id + 1))
                    l_style_total_history.append(l_style_total / (batch_id + 1))
                    l_total_total_history.append((l_feat_total + l_style_total) / (batch_id + 1))

                if (batch_id + 1) % self.args.log_interval == 0:
                    summary = "{}\tEpoch {}:\t[{}/{}]\tcontent: {:.6f}\tstyle: {:.6f}\ttotal: {:.6f}".format(
                        time.ctime(), epoch + 1, count, len(self.gen.train),
                        l_feat_total / (batch_id + 1),
                        l_style_total / (batch_id + 1),
                        (l_feat_total + l_style_total) / (batch_id + 1))
                    print(summary)

                if self.args.save:
                    if self.args.checkpoints_path is not None and (batch_id + 1) % self.args.checkpoints_interval == 0:
                        self.transformer.eval()
                        filename = "check_epoch_" + str(epoch) + "batch_id" + str(batch_id + 1) + '.pth'
                        path = os.path.join(self.args.checkpoints_path, filename)
                        print('\nSaving model:', path, '\n')
                        torch.save(self.transformer.state_dict(), path)
                        print('Model saved')
                        np.savetxt(self.args.checkpoints_path+"/losses/l_total_history.csv", l_total_history, delimiter=",")
                        np.savetxt(self.args.checkpoints_path+"/losses/l_feat_history.csv", l_feat_history, delimiter=",")
                        np.savetxt(self.args.checkpoints_path+"/losses/l_style_history.csv", l_style_history, delimiter=",")
                        np.savetxt(self.args.checkpoints_path+"/losses/l_total_total_history.csv", l_feat_total_history, delimiter=",")
                        np.savetxt(self.args.checkpoints_path+"/losses/l_feat_total_history.csv", l_feat_total_history, delimiter=",")
                        np.savetxt(self.args.checkpoints_path+"/losses/l_style_total_history.csv", l_style_total_history, delimiter=",")
                        print('Model losses saved.')

        self.transformer.eval()


def main():

    parser = argparse.ArgumentParser(description='Arguments for testing the network.')
    
    parser.add_argument('--data-dir', type=str, help='Path for training data', required=True)
    parser.add_argument('--style', type=str, help='Path for style image', required=True)
    parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')  

    parser.add_argument('--epochs', type=int, default=2, help='number of epochs')
    parser.add_argument('--batch', type=int, default=4, help='Batch size')
    
    parser.add_argument('--content-weights', type=int, default=1, help='Content weights')
    parser.add_argument('--style-weights', type=int, default=5, help='Style weights')
    parser.add_argument('--log-interval', type=int, default=1, help='Interval for logs')   
    
    parser.add_argument('--save', type=int, default=0, help='1 to save the checkpoints and 0 not to save')
    parser.add_argument('--checkpoints-path', type=str, help='Path for saving checkpoints')
    parser.add_argument('--checkpoints-interval', type=int, default=5, help='Interval for saving checkpoints')

    parser.add_argument('--retrain', type=int, default=0, help="1 for retraining a model and 0 for a new model")
    parser.add_argument('--retrain-model', type=str, default=None, help="Path for model to retrain")

    args = parser.parse_args()

    if args.retrain and (args.retrain_model is None):
        parser.error("--retrain requires --retrain-model.")

    if args.save and (args.checkpoints_path is None):
        parser.error("--save requires --checkpoints-path.")

    trainer = StyleTransferTrainer(args)
    trainer.train()

if __name__ == "__main__":
    main()