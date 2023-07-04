import torch
import torchvision
import matplotlib.pyplot as plt
from  d2l import torch as d2l


def preprocess_image(image, image_shape, img_mean, img_std):
    transform = torchvision.transforms.Compose([
        torchvision.transforms.Resize(image_shape),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean=img_mean, std=img_std)
    ])
    return transform(image).unsqueeze(0)

def postprocess(image, img_mean, img_std):
    img = image[0].to(img_mean.device)
    img = torch.clamp(img_std*img.permute(1, 2, 0) + img_mean, 0, 1)
    return torchvision.transforms.ToPILImage()(img.permute(2, 0, 1))

class SyntetizedImage(torch.nn.Module):
    def __init__(self, image_size, **kwargs):
        super().__init__(**kwargs)
        self.weight = torch.nn.Parameter(torch.normal(0, 1, image_size))
    def forward(self):
        return self.weight
def extract_features(X, content_layers, style_layers):
    contents = []
    styles = []
    X = X.to(device)
    for i in range(len(net)):
        X = net[i](X)
        if i in style_layers:
            styles.append(X)
        if i in content_layers:
            contents.append(X)
    return contents, styles

def gram_matrix(Y):
    num_channels, n = Y.shape[1], Y.numel()//Y.shape[1]
    Y = Y.reshape((num_channels, n))
    return torch.matmul(Y, Y.T)

def init_elements(content_image, style_image, image_shape, img_mean, img_std, net, content_layers, style_layers, device):
    content_image = preprocess_image(content_image, image_shape=image_shape, img_mean=img_mean, img_std=img_std)
    content_features, _ = extract_features(content_image, content_layers, style_layers)
    style_image = preprocess_image(style_image, image_shape=image_shape, img_mean=img_mean, img_std=img_std)
    _, style_features = extract_features(style_image, content_layers, style_layers)
    style_features = [gram_matrix(Y).to(device) for Y in style_features]

    content_image = content_image.to(device)
    style_image = style_image.to(device)


    return content_image, style_image, content_features, style_features

def style_loss(synt_img, style_features_gram):
    gram_mat = gram_matrix(synt_img)
    return torch.square(gram_matrix(synt_img) - style_features_gram.detach()).mean()

def content_loss(synt_image, content_features):
    return torch.square(synt_image - torch.tensor(content_features)).mean()

def tv_loss(synt_imgage):
    return torch.sum(torch.tensor([torch.abs(synt_imgage[:, :, 1:, :] - synt_imgage[:, :, :-1, :]).mean(),  torch.abs(synt_imgage[:, :, :, 1:] - synt_imgage[:, :, :, :-1]).mean()]))/2
def compute_loss(synthetized_image, contents_Y_hat, style_Y_hat, content_features, style_features_gram, loss_w_style, loss_w_content, loss_w_total):
    l_style = [loss_w_style*style_loss(preds, Y) for (preds, Y) in zip(style_Y_hat, style_features_gram)]
    l_contnet = [loss_w_content*content_loss(preds, Y) for (preds, Y) in zip(contents_Y_hat, content_features)]
    l_total_var_loss = loss_w_total*tv_loss(synthetized_image)
    l = sum(l_style+l_contnet + [l_total_var_loss])
    return l
def train(content_image, style_image, image_shape, img_mean, img_std, net, content_layers, style_layers, lr, num_epochs, device):
    content_image, style_image, content_features, style_features_gram = init_elements(content_image, style_image, image_shape, img_mean, img_std, net, content_layers, style_layers, device=device)

    synthetized_image = SyntetizedImage(content_image.shape)
    synthetized_image = synthetized_image.to(device)
    optimizer = torch.optim.Adam(synthetized_image.parameters(), lr=lr)

    synthetized_image.weight.data.copy_(content_image.data)
    synt_img = synthetized_image.forward()
    loss_metric = {}

    for epoch in range(num_epochs):
        print(f"Executing epoch {epoch+1}")
        optimizer.zero_grad()


        contents_Y_hat, style_Y_hat = extract_features(synt_img, content_layers, style_layers)
        loss = compute_loss(synt_img, contents_Y_hat, style_Y_hat, content_features, style_features_gram, 10**4, 1, 10)

        loss.backward()
        optimizer.step()
        loss_metric[f"loss_{epoch}"] = loss

        reconstruct_img = postprocess(synt_img, img_mean=img_mean, img_std=img_std)
        loss_metric[f"rec_img_{epoch}"] = reconstruct_img

    return loss_metric


if __name__ == "__main__":

    content_image = d2l.Image.open("./img/rainier.jpg")
    style_image = d2l.Image.open('./img/autumn-oak.jpg')

    content_layers = [2, 5, 9, 32]
    style_layers = [0, 5, 7, 11, 17, 19, 21, 28, 32, 35, ]
    # style_layers, content_layers = [0, 5, 10, 19, 28], [25]

    device = "cuda:0"
    vgg_pretrained = torchvision.models.vgg19(pretrained=True)
    net = torch.nn.Sequential(*[vgg_pretrained.features[i] for i in range(max(content_layers + style_layers) + 1)])
    net = net.to(device)


    image_shape = (300, 450)

    img_mean = torch.tensor([0.485, 0.456, 0.406])
    img_std = torch.tensor([0.229, 0.224, 0.225])

    lr = 0.005
    num_epochs = 200

    results = train(content_image, style_image, image_shape, img_mean, img_std, net, content_layers, style_layers, lr, num_epochs, device)

    plot_step_size = 20
    for x in range(0, len(results.keys())-plot_step_size-1, plot_step_size):
        plt.imshow(results[f"rec_img_{x}"])
        plt.pause(1)
        print(x)
