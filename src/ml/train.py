import mlflow
import wandb
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
)
from peft import LoraConfig, get_peft_model, TaskType
from datasets import load_dataset, load_from_disk
import yaml
from pathlib import Path
from src.config.settings import settings

def load_hyperparams(path: str = "src/ml/hyperparams.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)

def run_training():
    hp = load_hyperparams()
    train_cfg = hp["training"]

    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    mlflow.set_experiment(settings.experiment_name)
    wandb.init(project=settings.wandb_project, config=train_cfg)

    with mlflow.start_run(run_name="lora-finetune-llama31-8b-v2"):
        mlflow.log_params(train_cfg)
        mlflow.log_artifact("src/ml/hyperparams.yaml")

        tokenizer = AutoTokenizer.from_pretrained(train_cfg["model_name"])
        tokenizer.pad_token = tokenizer.eos_token

        model = AutoModelForCausalLM.from_pretrained(
            train_cfg["model_name"],
            torch_dtype="bfloat16",
            device_map="auto",
            token=settings.huggingface_token,
        )

        lora_config = LoraConfig(
            r=train_cfg["lora_r"],
            lora_alpha=train_cfg["lora_alpha"],
            lora_dropout=train_cfg["lora_dropout"],
            target_modules=train_cfg["target_modules"],
            task_type=TaskType.CAUSAL_LM,
        )
        model = get_peft_model(model, lora_config)

        dataset = load_dataset("parquet", data_files={
            "train": hp["data"]["train_file"],
            "validation": hp["data"]["val_file"],
        })

        training_args = TrainingArguments(
            output_dir="data/artifacts/lora-llama31-8b",
            num_train_epochs=train_cfg["epochs"],
            per_device_train_batch_size=train_cfg["batch_size"],
            gradient_accumulation_steps=train_cfg["gradient_accumulation_steps"],
            learning_rate=train_cfg["learning_rate"],
            lr_scheduler_type=train_cfg["lr_scheduler"],
            warmup_ratio=train_cfg["warmup_ratio"],
            weight_decay=train_cfg["weight_decay"],
            bf16=train_cfg["bf16"],
            gradient_checkpointing=train_cfg["gradient_checkpointing"],
            logging_steps=20,
            eval_strategy="steps",
            eval_steps=hp["evaluation"]["eval_steps"],
            save_steps=hp["evaluation"]["save_steps"],
            report_to=["mlflow", "wandb"],
            seed=train_cfg["seed"],
        )

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=dataset["train"],
            eval_dataset=dataset["validation"],
            data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
        )

        result = trainer.train()
        mlflow.log_metrics({
            "train_loss": result.training_loss,
            "train_runtime": result.metrics.get("train_runtime", 0),
        })

        model.save_pretrained("data/artifacts/lora-llama31-8b/final")
        tokenizer.save_pretrained("data/artifacts/lora-llama31-8b/final")
        mlflow.transformers.log_model(
            transformers_model={"model": model, "tokenizer": tokenizer},
            artifact_path="model",
            registered_model_name="rag-classifier",
        )
        wandb.finish()
        return result

if __name__ == "__main__":
    run_training()
