import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import re


class RAGModel:

    def __init__(self,model="huggyllama/llama-7b",tokenizer="huggyllama/llama-7b"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        print(f"Device to be used: {self.device}")

        self.oracle = pipeline("text-generation",
                               model=model,
                               tokenizer=tokenizer,
                               device=0,
                               torch_dtype=torch.float16,
                               return_full_text=False)

    def generate_answers(self, context, question):
        """
        It generates answers based on context.
        :param context: str, context to generate answers for
        :param question: str, question to generate answers for
        :return: str, the generated answers
        """
        prompt = (
            f"Context:\n{context}\n\n"
            f"Question:\n{question}\n\n"
            f"Give a clear and short answer. Do NOT repeat the question or the context."
            f"Answer:\n"
        )
        text = self.oracle(
            prompt,
            max_new_tokens=64,
            num_beams=3,
            early_stopping=True
        )[0]["generated_text"]
        answer = text.replace(prompt, '').strip()
        print(f"Answer: {answer}")
        text = re.sub(r'^(Answer:\s*)+', '', answer).strip()
        text = re.sub(r'\bQuestion:.*$', '', text).strip()
        text = re.split(r'(?<=[.!?])\s+', text)
        result = ' '.join(text[:2])
        print(f"Result: {result}")

        return result