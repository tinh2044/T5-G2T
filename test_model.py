import unittest
import torch
from transformers import T5Config
from models import GlossTextCLIP, T5_Model, TextCLIP  # Đảm bảo bạn có file `model.py`


class TestGlossTextCLIP(unittest.TestCase):
    """TestCase cho GlossTextCLIP"""
    
    @classmethod
    def setUpClass(cls):
        """Thiết lập mô hình chỉ một lần cho tất cả test cases"""
        cls.clip_model = GlossTextCLIP(config={"model": {"transformer": "t5-base"}}, task="clip")
        cls.g2t_model = GlossTextCLIP(config={"model": {"transformer": "t5-base"}}, task="g2t")

        cls.src_input = {
            "gloss_ids": torch.randint(0, 1000, (2, 10)),  
            "attention_mask": torch.ones((2, 10)),
            "labels": torch.randint(0, 1000, (2, 10)),
            "labels_attention_mask": torch.ones((2, 10))
        }

    def test_forward_clip_output_shape(self):
        """Kiểm tra shape của output khi chạy forward_clip"""
        logits_per_gloss, logits_per_text, ground_truth = self.clip_model(self.src_input)

        self.assertEqual(logits_per_gloss.shape, (2, 2), "logits_per_gloss sai shape")
        self.assertEqual(logits_per_text.shape, (2, 2), "logits_per_text sai shape")
        self.assertEqual(ground_truth.shape, (2, 2), "ground_truth sai shape")

    def test_forward_g2t_output_shape(self):
        """Kiểm tra output loss và logits khi chạy forward_g2t"""
        loss, logits = self.g2t_model(self.src_input)

        self.assertFalse(torch.isnan(loss), "Loss bị NaN")
        self.assertEqual(logits.shape, (2, 10, 32128), "Sai shape logits")

    def test_generate_output_shape(self):
        """Kiểm tra xem generate có trả về tensor đúng shape không"""
        out = self.g2t_model.generate(self.src_input, max_new_tokens=10, num_beams=3)

        self.assertIsInstance(out, torch.Tensor, "Output không phải tensor")
        self.assertEqual(out.shape[0], 2, "Batch size của output không đúng")
        self.assertLessEqual(out.shape[1], 10, "Chiều dài của output vượt quá max_new_tokens")


class TestT5Model(unittest.TestCase):
    """TestCase cho T5_Model"""

    @classmethod
    def setUpClass(cls):
        """Khởi tạo mô hình T5_Model"""
        cls.t5_model = T5_Model(config=T5Config.from_pretrained("t5-base"), lm_head_type="identity")

    def test_t5_model_identity_output_shape(self):
        """Kiểm tra output khi lm_head_type='identity'"""
        input_ids = torch.randint(0, 1000, (2, 10))
        attention_mask = torch.ones((2, 10))

        output, logits = self.t5_model(input_ids, attention_mask)

        self.assertEqual(output.shape, (2, 1024), "Sai shape của output lm_head")
        self.assertEqual(logits.shape, (2, 10, 32128), "Sai shape của logits")


class TestTextCLIP(unittest.TestCase):
    """TestCase cho TextCLIP"""

    @classmethod
    def setUpClass(cls):
        """Khởi tạo mô hình TextCLIP"""
        cls.textclip_model = TextCLIP(config=T5Config.from_pretrained("t5-base"), head_type="linear")

    def test_textclip_forward_output_shape(self):
        """Kiểm tra output shape của TextCLIP"""
        input_ids = torch.randint(0, 1000, (2, 10))
        attention_mask = torch.ones((2, 10))

        lm_head_out, txt_logits = self.textclip_model(input_ids, attention_mask)

        self.assertEqual(lm_head_out.shape, (2, 1024), "Sai shape của output lm_head")
        self.assertEqual(txt_logits.shape, (2, 10, 1024), "Sai shape của txt_logits")


if __name__ == '__main__':
    unittest.main()
