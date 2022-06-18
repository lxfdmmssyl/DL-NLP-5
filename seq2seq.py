import torch
from torch import nn
from torch import optim
import random
import time
import math
import pyinstrument
import torch.nn.functional as F
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
teacher_forcing_ratio = 0.5
MAX_LENGTH = 350
import pickle

with open('./sentences.pickle', 'rb') as f:
    datas = pickle.load(f)

from gensim.models import word2vec
model_cbow = word2vec.Word2Vec.load('./cbow.model')
mark = ['。”','？”','！”','。','？','！']
print('read model over')

class EncoderRNN(nn.Module):
    def __init__(self, hidden_size):
        super(EncoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.gru = nn.GRU(hidden_size, hidden_size)

    def forward(self, input, hidden):
        output, hidden = self.gru(torch.reshape(input, (1, 1, self.hidden_size)), hidden)
        return output, hidden

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=device)


class AttnDecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size, dropout_p=0.1, max_length=MAX_LENGTH):
        super(AttnDecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.dropout_p = dropout_p
        self.max_length = max_length

        self.attn = nn.Linear(self.hidden_size * 2, self.max_length)
        self.attn_combine = nn.Linear(self.hidden_size * 2, self.hidden_size)
        self.dropout = nn.Dropout(self.dropout_p)
        self.gru = nn.GRU(self.hidden_size, self.hidden_size)
        self.out = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, input, hidden, encoder_outputs):
        attn_weights = F.softmax(
            self.attn(torch.cat((torch.reshape(input, (1, 1, self.hidden_size))[0], hidden[0]), 1)), dim=1)
        attn_applied = torch.bmm(attn_weights.unsqueeze(0),
                                 encoder_outputs.unsqueeze(0))

        output = torch.cat((torch.reshape(input, (1, 1, self.hidden_size))[0], attn_applied[0]), 1)
        output = self.attn_combine(output).unsqueeze(0)

        output = F.relu(output)
        output, hidden = self.gru(output, hidden)

        # output = F.log_softmax(self.out(output[0]), dim=1)
        return output, hidden, attn_weights

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=device)


def train(input_tensor, target_tensor, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion, max_length=MAX_LENGTH):
    encoder_hidden = encoder.initHidden()

    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()

    input_length = input_tensor.size(0)
    target_length = target_tensor.size(0)

    encoder_outputs = torch.zeros(max_length, encoder.hidden_size, device=device)

    loss = 0

    for ei in range(input_length):
        encoder_output, encoder_hidden = encoder(
            input_tensor[ei], encoder_hidden)
        encoder_outputs[ei] = encoder_output[0, 0]

    decoder_input = encoder_outputs[-1]

    decoder_hidden = encoder_hidden

    use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False

    if use_teacher_forcing:
        # Teacher forcing: Feed the target as the next input
        for di in range(target_length):
            decoder_output, decoder_hidden, decoder_attention = decoder(
                decoder_input, decoder_hidden, encoder_outputs)
            loss += criterion(decoder_output, torch.reshape(target_tensor[di], (1,1,256)))
            decoder_input = target_tensor[di]  # Teacher forcing

    else:
        # Without teacher forcing: use its own predictions as the next input
        for di in range(target_length):
            decoder_output, decoder_hidden, decoder_attention = decoder(
                decoder_input, decoder_hidden, encoder_outputs)
            decoder_input = decoder_output  # detach from history as input

            loss += criterion(decoder_output, torch.reshape(target_tensor[di], (1,1,256)))
            '''
            dt = decoder_input.cpu()
            if model_cbow.wv.most_similar(positive=dt.detach().numpy().reshape((1, 256)), topn=1)[0][0] in mark:
                break
            '''

    loss.backward()

    encoder_optimizer.step()
    decoder_optimizer.step()

    return loss.item() / target_length

import matplotlib.pyplot as plt
plt.switch_backend('agg')
import matplotlib.ticker as ticker
import numpy as np

def asMinutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


def timeSince(since, percent):
    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s
    return '%s (- %s)' % (asMinutes(s), asMinutes(rs))

def showPlot(points):
    plt.figure()
    fig, ax = plt.subplots()
    # this locator puts ticks at regular intervals
    loc = ticker.MultipleLocator(base=0.2)
    ax.yaxis.set_major_locator(loc)
    plt.plot(points)

def trainIters(encoder, decoder, n_iters, print_every=1000, plot_every=100, learning_rate=0.01):
    start = time.time()
    plot_losses = []
    print_loss_total = 0  # Reset every print_every
    plot_loss_total = 0  # Reset every plot_every

    encoder_optimizer = optim.SGD(encoder.parameters(), lr=learning_rate)
    decoder_optimizer = optim.SGD(decoder.parameters(), lr=learning_rate)
    training_pairs = [tensorsFromSentences()
                      for i in range(n_iters)]
    # TODO
    criterion = nn.MSELoss()

    for iter in range(1, n_iters + 1):
        training_pair = training_pairs[iter - 1]
        input_tensor = torch.Tensor(training_pair[0]).cuda()
        target_tensor = torch.Tensor(training_pair[1]).cuda()

        loss = train(input_tensor, target_tensor, encoder,
                     decoder, encoder_optimizer, decoder_optimizer, criterion)
        print_loss_total += loss
        plot_loss_total += loss

        if iter % print_every == 0:
            print_loss_avg = print_loss_total / print_every
            print_loss_total = 0
            print('%s (%d %d%%) %.4f' % (timeSince(start, iter / n_iters),
                                         iter, iter / n_iters * 100, print_loss_avg))
            torch.save(encoder, "./model/encoder{}".format(iter))
            torch.save(decoder, "./model/decoder{}".format(iter))

        if iter % plot_every == 0:
            plot_loss_avg = plot_loss_total / plot_every
            plot_losses.append(plot_loss_avg)
            plot_loss_total = 0

    showPlot(plot_losses)

def tensorsFromSentences():
    input = []
    target = []
    i = random.randint(0, len(datas)-1)
    j = random.randint(0, len(datas[i])-2)
    for word in datas[i][j]:
        input.append(np.float32(model_cbow.wv[word]))
    for word in datas[i][j+1]:
        target.append(np.float32(model_cbow.wv[word]))
    return (input, target)

if __name__ == "__main__":
    ''' # train
    hidden_size = 256
    output_size = 256
    encoder1 = EncoderRNN(hidden_size).to(device)
    attn_decoder1 = AttnDecoderRNN(hidden_size, output_size, dropout_p=0.1).to(device)

    trainIters(encoder1, attn_decoder1, 20000, print_every=5000)
    '''

    encoder = torch.load('./model/encoder20000')
    decoder = torch.load('./model/decoder20000')
    print('read en/de model over')

    index = 3
    sen = 101
    for i in range(sen, sen+20):
        for j in datas[index][i]:
            print(j,end="")
        print()

    for i in range(sen, sen+6):
        with torch.no_grad():
            input = []
            for word in datas[index][i]:
                input.append(np.float32(model_cbow.wv[word]))

    encoder_hidden = encoder.initHidden()
    encoder_outputs = torch.zeros(MAX_LENGTH, encoder.hidden_size, device=device)
    t = 0

    for ei in range(len(input)):
        t = ei
        encoder_output, encoder_hidden = encoder(
            torch.Tensor(input[ei]).cuda(), encoder_hidden)
        encoder_outputs[ei] = encoder_output[0, 0]

    decoder_input = encoder_outputs[t]

    decoder_hidden = encoder_hidden

    outputs = []
    length = 0
    sentences = 0
    while length < 1000 and sentences < 10:
        decoder_output, decoder_hidden, decoder_attention = decoder(
            decoder_input, decoder_hidden, encoder_outputs)
        decoder_input = decoder_output  # detach from history as input

        dt = decoder_input.cpu()
        word = model_cbow.wv.most_similar(positive=dt.detach().numpy().reshape((1, 256)), topn=10)[random.randint(0,9)][0]
        outputs.append(word)
        decoder_input = torch.reshape(torch.Tensor(model_cbow.wv[word]), (1, 1, 256)).cuda()
        length = length + 1
        if word in mark:
            sentences = sentences + 1

    print(outputs)
    for i in outputs:
        print(i, end="")

