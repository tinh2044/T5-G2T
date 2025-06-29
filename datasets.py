import utils as utils
import torch.utils.data.dataset as Dataset
import random
import numpy as np
import pandas as pd


class G2T_Dataset(Dataset.Dataset):
    def __init__(self, path, tokenizer, config, args, phase, training_refurbish=False):
        self.config = config
        self.args = args
        self.training_refurbish = training_refurbish

        self.tokenizer = tokenizer
        self.phase = phase

        csv_path = f"{path}/{phase}.csv"
        self.df = pd.read_csv(csv_path, sep="|")

    def random_deletion(self, sentence, prob=0.1):
        words = sentence.split()
        if len(words) == 1:
            return sentence
        return " ".join([word for word in words if random.random() > prob])

    def random_insertion(self, sentence, prob=0.1):
        words = sentence.split()
        n = len(words)
        for _ in range(int(prob * n)):
            idx = random.randint(0, n - 1)
            insert_idx = random.randint(0, n)
            words.insert(insert_idx, words[idx])
        return " ".join(words)

    def add_noise(self, sentence, prob=0.05):
        def swap_chars(word):
            if len(word) > 1:
                idx = random.randint(0, len(word) - 2)
                return word[:idx] + word[idx + 1] + word[idx] + word[idx + 2 :]
            return word

        words = sentence.split()
        noisy_words = [
            swap_chars(word) if random.random() < prob else word for word in words
        ]
        return " ".join(noisy_words)

    def augment_data(self, text):
        aug = False
        while not aug:
            if np.random.uniform(0.0, 1.0) > 0.5:
                text = self.random_deletion(text, np.random.uniform(0.1, 0.5))
                aug = True
            if np.random.uniform(0.0, 1.0) > 0.5:
                text = self.random_insertion(text, np.random.uniform(0.1, 0.5))
                aug = True
            if np.random.uniform(0.0, 1.0) > 0.5:
                text = self.add_noise(text, np.random.uniform(0.1, 0.5))
                aug = True
        return text

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        row = self.df.iloc[index]
        name = row["name"]
        gloss = row["orth"]
        text = row["translation"]

        return name, gloss, text

    def collate_fn(self, batch):
        name_batch, gloss_batch, text_batch = zip(*batch)
        outputs = {}

        gloss_batch = self.process_gloss(gloss_batch)

        text_output = self.tokenizer(
            text_batch, return_tensors="pt", padding=True, truncation=True
        )
        max_text_length = text_output["input_ids"].shape[1]

        gloss_output = self.tokenizer(
            gloss_batch,
            return_tensors="pt",
            padding="max_length",
            max_length=max_text_length,
            truncation=True,
        )

        outputs["gloss_ids"] = gloss_output["input_ids"]
        outputs["attention_mask"] = gloss_output["attention_mask"]
        outputs["labels_attention_mask"] = text_output["attention_mask"]

        labels = text_output["input_ids"]
        outputs["labels"] = labels

        outputs["gloss_inputs"] = gloss_batch
        outputs["text_inputs"] = text_batch
        outputs["name_batch"] = name_batch

        return outputs

    def process_gloss(self, gloss_batch):
        process_gloss_batch = []

        for gloss in gloss_batch:
            if self.phase == "train" and np.random.uniform(0, 1) < 0.5:
                process_gloss_batch.append(self.augment_data(gloss))
            else:
                process_gloss_batch.append(gloss)

        return process_gloss_batch

    def __str__(self):
        info = f"""
        Dataset Information:
        -----------------------
        - Phase: {self.phase}
        - Total samples: {len(self.df)}
        - Data Augmentation: {"Enabled" if self.phase == "train" else "Disabled"}
        - Tokenizer: {self.tokenizer.name_or_path}
        - Example Samples:
            - Gloss: {self.df.iloc[0]["orth"]}
            - Text: {self.df.iloc[0]["translation"]}
        """
        return info.strip()


if __name__ == "__main__":
    from transformers import T5Tokenizer
    from torch.utils.data import DataLoader

    tokenizer = T5Tokenizer.from_pretrained("google-t5/t5-base")
    dataset = G2T_Dataset("./data/Phonexi-2014T", tokenizer, None, None, "train")
    loader = DataLoader(dataset, batch_size=4, collate_fn=dataset.collate_fn)
    for item in loader:
        decodes_gloss = dataset.tokenizer.batch_decode(
            item["gloss_ids"], skip_special_tokens=True
        )
        decodes_text = dataset.tokenizer.batch_decode(
            item["labels"], skip_special_tokens=True
        )

        for d_t, d_g, g, t in zip(
            decodes_text, decodes_gloss, item["gloss_inputs"], item["text_inputs"]
        ):
            assert d_t == t, print(f"{d_t} != {t}")
            assert d_g == g, print(f"{d_g} != {g}")
