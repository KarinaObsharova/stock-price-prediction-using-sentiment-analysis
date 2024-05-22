from dostoevsky.tokenization import RegexTokenizer
from dostoevsky.models import FastTextSocialNetworkModel
import json
import fasttext

class DostoevskyTrainer:
    def __init__(self, data_path, model_save_path, pretrained_model_path=None):
        self.data_path = data_path
        self.model_save_path = model_save_path
        self.pretrained_model_path = pretrained_model_path

    def load_data(self):
        with open(self.data_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return [(item['text'], item['sentiment']) for item in data]

    def prepare_fasttext_data(self, data):
        with open('fasttext_training_data.txt', 'w', encoding='utf-8') as f:
            for text, sentiment in data:
                label = f'__label__{sentiment}'
                f.write(f'{label} {text}\n')

    def train_fasttext_model(self):
        if self.pretrained_model_path:
            model = fasttext.train_supervised(
                input='fasttext_training_data.txt',
                pretrainedVectors=self.pretrained_model_path,
                epoch=10,
                lr=0.1,
                wordNgrams=2,
                dim=300
            )
        else:
            model = fasttext.train_supervised(
                input='fasttext_training_data.txt',
                epoch=10,
                lr=0.1,
                wordNgrams=2,
                dim=300
            )
        model.save_model(self.model_save_path)
        return model

    def test_model(self, model, data):
        correct = 0
        total = 0
        for text, sentiment in data:
            label = f'__label__{sentiment}'
            text = text.replace('\n', '')
            prediction = model.predict(text)
            if prediction[0][0] == label:
                correct += 1
            total += 1
        print(f'Accuracy: {correct / total:.2f}')

    def train(self):
        data = self.load_data()
        self.prepare_fasttext_data(data)
        model = self.train_fasttext_model()
        self.test_model(model, data)

    def save_model_for_dostoevsky(self):
        tokenizer = RegexTokenizer()
        model = FastTextSocialNetworkModel.load_model(model_path=self.model_save_path, tokenizer=tokenizer)
        model.save(self.model_save_path.replace('.bin', '.dostoevsky'))

if __name__ == "__main__":
    trainer = DostoevskyTrainer(
        data_path='/Users/malikarii/PycharmProjects/last-work/sentiment-analysis/train.json',
        model_save_path='/Users/malikarii/PycharmProjects/last-work/service_aggregator/model_dostoevsky.bin',
    )
    trainer.train()
    trainer.save_model_for_dostoevsky()
