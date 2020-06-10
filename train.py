import os
import math
import random
import logging
import warnings
import collections
import wandb
from dict_to_obj import DictToObj
from timeit import default_timer as timer

warnings.filterwarnings("ignore")
wandb.login()

from transformers import (
    MODEL_WITH_LM_HEAD_MAPPING,
    AutoConfig,
    AutoModelWithLMHead,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    LineByLineTextDataset,
    TextDataset,
    Trainer,
    TrainingArguments,
    set_seed,
)

start = "<|startoftext|>"
sep = "<|sep|>"


def get_dataset(args, tokenizer, evaluate=False):
    file_path = args.eval_data_file if evaluate else args.train_data_file
    if args.line_by_line:
        return LineByLineTextDataset(
            tokenizer=tokenizer, file_path=file_path, block_size=args.block_size
        )
    else:
        return TextDataset(
            tokenizer=tokenizer,
            file_path=file_path,
            block_size=args.block_size,
            overwrite_cache=args.overwrite_cache,
        )


# Logging
logger = logging.getLogger(__name__)
# Model classes
MODEL_CONFIG_CLASSES = list(MODEL_WITH_LM_HEAD_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)

# These arguments could have been handled by CLI, but I put them in this
# way to make the code simpler.

# Model arguments
model_args = collections.defaultdict(
    config_name="gpt2",
    model_name_or_path="gpt2-large",
    model_type="gpt2",
    tokenizer_name="gpt2",
    cache_dir=None,
)
# Data arguments
data_args = collections.defaultdict(
    train_data_file="data/train.txt",
    eval_data_file="data/valid.txt",
    line_by_line=False,
    mlm=False,
    mlm_probability=0.15,
    block_size=512,
    overwrite_cache=False,
)
# Training arguments
training_args = TrainingArguments(
    output_dir="/model",
    overwrite_output_dir=True,
    do_train=True,
    do_eval=True,
    do_predict=False,
    evaluate_during_training=True,
    per_gpu_train_batch_size=1,
    per_gpu_eval_batch_size=1,
    gradient_accumulation_steps=1,
    learning_rate=1e-5,
    weight_decay=0.0,
    adam_epsilon=1e-08,
    max_grad_norm=1.0,
    num_train_epochs=5.0,
    max_steps=-1,
    warmup_steps=0,
    logging_dir=None,
    logging_first_step=False,
    logging_steps=1000,
    save_steps=10000,
    save_total_limit=100000,
    no_cuda=False,
    seed=42,
    fp16=False,
    fp16_opt_level="O1",
    local_rank=-1,
)
# Convert dict to objects
model_args = DictToObj(model_args)
data_args = DictToObj(data_args)

# Logging
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO if training_args.local_rank in [-1, 0] else logging.WARN,
)
logger.warning(
    "Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
    training_args.local_rank,
    training_args.device,
    training_args.n_gpu,
    bool(training_args.local_rank != -1),
    training_args.fp16,
)
logger.info("Training/evaluation parameters %s", training_args)

# Seed
set_seed(training_args.seed)

# Load tokenizer and model
config = AutoConfig.from_pretrained(
    model_args.model_name_or_path, cache_dir=model_args.cache_dir
)
tokenizer = AutoTokenizer.from_pretrained(
    model_args.model_name_or_path, cache_dir=model_args.cache_dir
)
model = AutoModelWithLMHead.from_pretrained(
    model_args.model_name_or_path,
    from_tf=bool(".ckpt" in model_args.model_name_or_path),
    config=config,
    cache_dir=model_args.cache_dir,
)

# Add special tokens
tokenizer.add_special_tokens({"sep_token": sep})
tokenizer.add_special_tokens({"bos_token": start})
model.resize_token_embeddings(len(tokenizer))

# Load dataset
train_dataset = (
    get_dataset(data_args, tokenizer=tokenizer) if training_args.do_train else None
)

eval_dataset = (
    get_dataset(data_args, tokenizer=tokenizer, evaluate=True)
    if training_args.do_eval
    else None
)

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer, mlm=data_args.mlm, mlm_probability=data_args.mlm_probability,
)

# Initialize trainer
trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    prediction_loss_only=True,
)

# Define model path
model_path = (
    model_args.model_name_or_path
    if model_args.model_name_or_path is not None
    and os.path.isdir(model_args.model_name_or_path)
    else None
)

# Train the model
start = timer()
train_results = trainer.train(model_path=model_path)
end = timer()
trainer.save_model()
if trainer.is_world_master():
    tokenizer.save_pretrained(training_args.output_dir)

# Calculate training time
logger.info(f"Training took {(end - start) / 3600} hours.")


# Evaluation on validation set
logger.info("*** Valid Evaluate ***")
valid_eval_output = trainer.evaluate()
valid_perplexity = math.exp(valid_eval_output["eval_loss"])
valid_result = {"valid_perplexity": valid_perplexity}
output_eval_file = os.path.join(training_args.output_dir, "valid_eval_results_lm.txt")

with open(output_eval_file, "w") as writer:
    logger.info("***** Valid Eval results *****")
    for key in sorted(valid_result.keys()):
        logger.info("  %s = %s", key, str(valid_result[key]))
        writer.write("%s = %s\n" % (key, str(valid_result[key])))


# Evaluation on test set
training_args.do_eval = True
data_args.eval_data_file = "data/test.txt"
test_dataset = (
    get_dataset(data_args, tokenizer=tokenizer, evaluate=True)
    if training_args.do_eval
    else None
)
trainer.eval_dataset = test_dataset

logger.info("*** Test Evaluate ***")
test_eval_output = trainer.evaluate()
test_perplexity = math.exp(test_eval_output["eval_loss"])
test_result = {"test_perplexity": test_perplexity}
output_eval_file = os.path.join(training_args.output_dir, "test_eval_results_lm.txt")

with open(output_eval_file, "w") as writer:
    logger.info("***** Test Eval results *****")
    for key in sorted(test_result.keys()):
        logger.info("  %s = %s", key, str(test_result[key]))
        writer.write("%s = %s\n" % (key, str(test_result[key])))


# Evaluation on training set
data_args.eval_data_file = "data/train.txt"
test_dataset = (
    get_dataset(data_args, tokenizer=tokenizer, evaluate=True)
    if training_args.do_eval
    else None
)
trainer.eval_dataset = test_dataset

logger.info("*** Train Evaluate ***")
train_eval_output = trainer.evaluate()
train_perplexity = math.exp(train_eval_output["eval_loss"])
train_result = {"train_perplexity": train_perplexity}
output_eval_file = os.path.join(training_args.output_dir, "train_eval_results_lm.txt")

with open(output_eval_file, "w") as writer:
    logger.info("***** Train Eval results *****")
    for key in sorted(train_result.keys()):
        logger.info("  %s = %s", key, str(train_result[key]))
        writer.write("%s = %s\n" % (key, str(train_result[key])))


print(f"Train loss: {train_eval_output['eval_loss']}")
print(f"Valid loss: {valid_eval_output['eval_loss']}")
print(f"Test loss: {test_eval_output['eval_loss']}")
print(f"Train PPL: {train_perplexity}")
print(f"Valid PPL: {valid_perplexity}")
print(f"Test PPL: {test_perplexity}")
