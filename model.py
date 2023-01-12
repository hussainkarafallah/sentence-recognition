import torch.nn as nn
import torch


class OCRModel(nn.Module):

    def __init__(self, num_classes=1, num_lstms=1) -> None:
        super(OCRModel, self).__init__()
        self.num_classes = num_classes
        self.num_lstms = num_lstms
        self.convs = nn.Sequential(
            # first block
            nn.Conv2d(1, 64, kernel_size=(3, 3), stride=(1,2)),
            #nn.BatchNorm2d(64),
            nn.LeakyReLU(),
            nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(1,2)),
            #nn.BatchNorm2d(128),
            nn.LeakyReLU(),
            nn.Conv2d(64, 128, kernel_size=(3, 3), stride=(1,1), padding='same'),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(),
            nn.Conv2d(128, 128, kernel_size=(3, 3), stride=(1,1), padding='same'),
            #nn.BatchNorm2d(256),
            nn.LeakyReLU(),
            nn.MaxPool2d((2, 2), stride=(2, 2)),

            nn.Conv2d(128, 256, kernel_size=(3, 3), stride=(1,1), padding='same'),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(),
            nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1,1), padding='same'),
            #nn.BatchNorm2d(512),
            nn.LeakyReLU(),
            nn.MaxPool2d((2, 2), stride=(2, 1)),

            nn.Conv2d(256, 256, kernel_size=(2, 2), stride=(1,1), padding='same'),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(),
        )
        self.lstm = nn.LSTM(
            input_size=256, hidden_size=128,
            num_layers=self.num_lstms,
            batch_first=True, bidirectional=True
        )
        self.dense = nn.Sequential(
            nn.Linear(128*2, 512),
            nn.LeakyReLU(),
            nn.Linear(512, 512),
            nn.LeakyReLU(),
            nn.Linear(512, self.num_classes + 1)
        )

    def forward(self, x):
        # x has shape (B-size, h, w, c)
        # extract visual features
        conv_feat = torch.permute(x, (0, 3, 1, 2))
        # now conv_feat has shape (B-size, c, h, w)
        conv_feat = self.convs(conv_feat)

        # remove h from shape
        conv_feat = nn.MaxPool2d((conv_feat.shape[2], 1))(conv_feat)
        # now conv_feat has shape (B-size, c, 1, w)
        conv_feat = torch.permute(conv_feat, (0, 3, 2, 1))
        # now conv_feat has shape (B-size, w, 1, c)

        # reshape
        new_shape = (
            conv_feat.shape[0],
            conv_feat.shape[1],
            -1 # merge the last 2 dims
        )
        conv_feat = torch.reshape(conv_feat, (new_shape))
        # now conv_feat has shape (B-size, w, c)

        # extract time related features
        lstm_out, _ = self.lstm(conv_feat)

        logits = self.dense(lstm_out)
        # logits have shape (B-size, w, num_classes + 1)

        #logits = torch.permute(logits, (1, 0, 2))

        log_probs = nn.functional.log_softmax(logits, dim=-1)

        return log_probs
