from typing import Any
from imports import *

class Classifier(torch.nn.Module):
    def __init__(self):
        super().__init__()
        
    def train_step(self, batch):
        y_pred = self.forward(*batch[:-1])
        l, acc = self.loss(y_pred, batch[-1])
        return l, acc

    def valid_step(self, batch):
        y_pred = self.forward(*batch[:-1])
        l, acc = self.loss(y_pred, batch[-1])
        return l, acc
    
    def loss(self, y_pred, y_true):
        raise NotImplementedError("Implement the loss for the classifier. ")
    
    def configure_optimizer(self, ):
        raise NotImplementedError("Implement the loss for the classifier. ")

class ConvBlock(torch.nn.Module):
    def __init__(self, num_chanels, use_1x1_conv):
        super().__init__()
        self.CNN1 = torch.nn.LazyConv2d(num_chanels, kernel_size=3, padding=1, stride=1)
        self.BN1 = torch.nn.LazyBatchNorm2d()
        self.relu1 = torch.nn.ReLU()
        self.CNN2 = torch.nn.LazyConv2d(num_chanels, kernel_size=3, padding=1, stride=1)
        self.BN2 = torch.nn.LazyBatchNorm2d()        
        self.use_1x1_conv = use_1x1_conv
        if use_1x1_conv==True:
            self.CNN3 = torch.nn.LazyConv2d(num_chanels, kernel_size=1, padding=0, stride=1)
        self.pool = torch.nn.MaxPool2d((3, 3))
    
    def forward(self, X):        
        out1 = self.relu1(self.BN1(self.CNN1(X)))
        out2 = self.BN2(self.CNN2(out1))
        if self.use_1x1_conv:
            X = self.CNN3(X)
        out2 += X
        return self.pool(out2)


    def init_network(self, X):
        out = self.forward(X)
        for name, param in self.named_parameters():
            if "BN" in name:
                continue
            if "weight" in name:
                torch.nn.init.kaiming_normal_(param)
            if "bias" in name:
                torch.nn.init.zeros_(param)
        return out

class InputPrep(torch.nn.Module):
    def __init__(self, ):
        super().__init__()
        self.conv_layer = torch.nn.LazyConv2d(64, kernel_size=7, stride=2, padding=3)
        self.bnlayer = torch.nn.LazyBatchNorm2d()
        self.relu = torch.nn.ReLU()
        self.max_pool = torch.nn.MaxPool2d(kernel_size=2, padding=1, stride=1)

    def forward(self, X):
        X = self.conv_layer(X)
        X = self.bnlayer(X)
        X = self.relu(X)
        X = self.max_pool(X)
        return X

    def init_network(self, X):
        out = self.forward(X)
        for name, param in self.named_parameters():
            if "bnlayer" in name:
                continue
            if "weight" in name:
                torch.nn.init.kaiming_normal_(param)
            if "bias" in name:
                torch.nn.init.zeros_(param)
        return out

class HeadLayer(torch.nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.pool_layer = torch.nn.AdaptiveAvgPool2d((1, 1))
        self.flatten_layer = torch.nn.Flatten()
        self.output_layer = torch.nn.LazyLinear(num_classes)
    
    def forward(self, X ):
        return self.output_layer(self.flatten_layer(self.pool_layer(X)))
    
    def init_network(self, X):
        out = self.forward(X)
        for name, param in self.named_parameters():
            if "weight" in name:
                torch.nn.init.kaiming_normal_(param)
            if "bias" in name:
                torch.nn.init.zeros_(param)
        return out

class CNNClassifier(Classifier):
    def __init__(self, lr, num_chanels, num_classes):
        super().__init__()
        self.lr = lr
        self.input_prep = InputPrep()
        self.res_block_1 = ConvBlock(num_chanels, True)
        self.res_block_2 = ConvBlock(num_chanels, False)
        self.head = HeadLayer(num_classes=num_classes)
      
    def forward(self, X):
        X = self.input_prep(X)
        X = self.res_block_1(X)
        X = self.res_block_2(X)
        return self.head(X)

    def loss(self, y_pred, y_true):
        l = torch.nn.CrossEntropyLoss()
        loss_value = l(input=y_pred, target=y_true)
        y_predicts = torch.argmax(y_pred, axis=1)
        acc = torch.mean((y_predicts == y_true).float())
        return loss_value, acc
    

    def init_network(self, X):
        output1 = self.input_prep.init_network(X)
        output2 = self.res_block_1.init_network(output1)
        output3 = self.res_block_2.init_network(output2)
        output5 = self.head.init_network(output3)

    def configure_optimizer(self,):
        return torch.optim.Adam(self.parameters(), lr=self.lr, betas=[0.9, 0.99])
    












class DoubleConv(torch.nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None, residual=False) -> None:
        super().__init__()
        self.residual = residual

        if not mid_channels:
            mid_channels = out_channels
        
        self.double_conv = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            torch.nn.GroupNorm(1, mid_channels),
            torch.nn.GELU(), 
            torch.nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            torch.nn.GroupNorm(1, out_channels)
        )

    
    def forward(self, x):
        if self.residual:
            return torch.nn.functional.gelu(x + self.double_conv(x))
        else:
            return self.double_conv(x)


class Down(torch.nn.Module):
    def __init__(self, in_channels, out_channels, emb_dim=128):
        super().__init__()

        self.maxpool_conv = torch.nn.Sequential(
            torch.nn.MaxPool2d(2),
            DoubleConv(in_channels, in_channels, residual=True),
            DoubleConv(in_channels, out_channels)
        )

        self.emb_layer = torch.nn.Sequential(
            torch.nn.SiLU(),
            torch.nn.Linear(
                emb_dim, 
                out_channels
            )
        )  # This layer is used for embedding the time dimensions of the step when the noise was injected. 

    def forward(self, x, t):
        x = self.maxpool_conv(x)
        emb = self.emb_layer(t)[:, :, None, None].repeat(1, 1, x.shape[-2], x.shape[-1])
        return x + emb
    

class Up(torch.nn.Module):
    def __init__(self, in_channels, out_channels, emb_dim=128):
        super().__init__()

        self.up = torch.nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)

        self.conv = torch.nn.Sequential(
            DoubleConv(in_channels, in_channels, residual=True),
            DoubleConv(in_channels, out_channels, in_channels//2)
        )

        self.emb_layer = torch.nn.Sequential(
            torch.nn.SiLU(),
            torch.nn.Linear(
                emb_dim, 
                out_channels
            )
        )  # This layer is used for embedding the time dimensions of the step when the noise was injected. 

    def forward(self, x, skip_x, t):
        x = self.up(x)
        x = torch.cat([skip_x, x], dim=1)
        x = self.conv(x)
        emb = self.emb_layer(t)[:, :, None, None].repeat(1, 1, x.shape[-2], x.shape[-1])
        return x + emb



class SelfAttention(torch.nn.Module):
    def __init__(self, channels, size):
        super(SelfAttention, self).__init__()
        self.size = size
        self.channels = channels
        self.mha = torch.nn.MultiheadAttention(channels, 4, batch_first=True)
        self.ln = torch.nn.LayerNorm([channels])

        self.ff_self = torch.nn.Sequential(
            torch.nn.LayerNorm([channels]),
            torch.nn.Linear(channels, channels),
            torch.nn.GELU(), 
            torch.nn.Linear(channels, channels),
        )

    def forward(self, x):
        x = x.view(-1, self.channels, self.size*self.size).swapaxes(1, 2)
        x_ln = self.ln(x)
        attention_value, _ = self.mha(x_ln, x_ln, x_ln)
        attention_value = attention_value + x
        attention_value = self.ff_self(attention_value) + attention_value
        out = attention_value.swapaxes(2, 1).view(-1, self.channels, self.size, self.size)
        return out



class UNet(torch.nn.Module):
    def __init__(self, c_in, c_out, lr, time_dim=128, device="cuda:0") -> None:
        super().__init__()
        self.device = device
        self.time_dim = time_dim
        self.lr = lr

        self.inc = DoubleConv(c_in, 64)
        self.down1 = Down(64, 128) # (input output)
        self.sa1 = SelfAttention(128, 32) # (chanel dimension, image resolution)
        self.down2 = Down(128, 256)
        self.sa2 = SelfAttention(256, 16)
        self.down3 = Down(256, 256)
        self.sa3 = SelfAttention(256, 8)



        self.bot1 = DoubleConv(256, 512)
        self.bot2 = DoubleConv(512, 512)
        self.bot3 = DoubleConv(512, 256)


        self.up1 = Up(512, 128)
        self.sa4 = SelfAttention(128, 16)
        self.up2 = Up(256, 64)
        self.sa5 = SelfAttention(64, 32)
        self.up3 = Up(128, 64)
        self.sa6 = SelfAttention(64, 64)

        self.outc = torch.nn.Conv2d(64, c_out, kernel_size=1)


    def pos_encodings(self, t, chanels):
        inv_freq = 1.0/ (10000**(torch.arange(0, chanels, 2, device= self.device).float())/chanels)
        pos_enc_a = torch.sin(t.repeat(1, chanels//2)*inv_freq)
        pos_enc_b = torch.cos(t.repeat(1, chanels//2)*inv_freq)
        pos_enc = torch.cat([pos_enc_a, pos_enc_b], dim=-1)
        return pos_enc


    def forward(self, x, t):
        
        t = t.unsqueeze(-1).type(torch.float)
        t = self.pos_encodings(t, self.time_dim)

        x1 = self.inc(x)
        x2 = self.down1(x1, t)
        x2 = self.sa1(x2)
        x3 = self.down2(x2, t)
        x3 = self.sa2(x3)
                        
        x4 = self.down3(x3, t)
        x4 = self.sa3(x4)
        
        x4 = self.bot1(x4)
        x4 = self.bot2(x4)
        x4 = self.bot3(x4)
        
        x = self.up1(x4, x3, t)
        x = self.sa4(x)
        x = self.up2(x, x2, t)
        
        x = self.sa5(x)
        x = self.up3(x, x1, t)
        x = self.sa6(x)

        output = self.outc(x)
        
        return output



    def configure_optimizer(self):
        return torch.optim.AdamW(self.parameters(), lr=self.lr)


    def loss(self, y_pred, y):
        l = torch.nn.MSELoss()
        return l(y_pred, y)