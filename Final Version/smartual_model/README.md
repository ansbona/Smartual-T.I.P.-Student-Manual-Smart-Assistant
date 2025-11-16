---
tags:
- sentence-transformers
- sentence-similarity
- feature-extraction
- dense
- generated_from_trainer
- dataset_size:20000
- loss:MultipleNegativesRankingLoss
base_model: sentence-transformers/all-MiniLM-L12-v2
widget:
- source_sentence: Rules regarding Hearing procedures and due process (policy)
  sentences:
  - Graduation requires fulfillment of all academic requirements, clearance of financial
    obligations, completion of residency, and submission of necessary documents as
    listed in the graduation checklist.
  - Students charged with offenses are entitled to notice, access to evidence, the
    opportunity to answer, and representation during hearings. Hearing committees
    are constituted according to the Manual to ensure impartiality.
  - Student organizations must secure registration and recognition from the Office
    of Student Affairs. Official representation in external activities requires written
    authorization and compliance with OSA procedures.
- source_sentence: Procedure for Use of campus facilities and hours
  sentences:
  - An NC is permanent and assigned when a student fails to take required examinations
    or submit essential academic requirements; it carries no credit and cannot be
    converted to a passing grade.
  - Scholarship eligibility depends on academic performance and other criteria; renewal
    requires compliance with the conditions stated in the scholarship guidelines and
    submission of required documents.
  - Access to campus facilities is subject to posted hours of operation and may require
    prior booking or authorization. Misuse or vandalism of facilities results in disciplinary
    measures.
- source_sentence: Official rule on Incomplete (4.0) and removal
  sentences:
  - Use of laboratories requires compliance with safety protocols, proper attire,
    and authorization from the laboratory-in-charge. Unauthorized access and misuse
    of equipment will result in disciplinary action.
  - Use of laboratories requires compliance with safety protocols, proper attire,
    and authorization from the laboratory-in-charge. Unauthorized access and misuse
    of equipment will result in disciplinary action.
  - An INC (4.0) must be removed within one year by completing outstanding requirements;
    failure to remove the INC within the allowed period converts it to a failing grade
    (5.0).
- source_sentence: Rules regarding Special examination policy
  sentences:
  - Graduation requires fulfillment of all academic requirements, clearance of financial
    obligations, completion of residency, and submission of necessary documents as
    listed in the graduation checklist.
  - Special examinations may be requested within the timeframe specified in the manual
    and require payment of the applicable special exam fee, clearance of financial
    obligations, and approval from the Program Chair.
  - Scholarship eligibility depends on academic performance and other criteria; renewal
    requires compliance with the conditions stated in the scholarship guidelines and
    submission of required documents.
- source_sentence: Prohibited clothing on non-uniform days policy (policy)
  sentences:
  - Students must complete Physical Education and NSTP/CWTS within the prescribed
    curriculum period. Failure to complete these may delay graduation or limit enrollment
    options as detailed in the Manual.
  - 'Final grades are computed using the progressive formulas in the manual: Period
    Grade (PG), Midterm Grade (MG), and Final Grade (FG) which combine class standing
    and examination components according to prescribed weights.'
  - Indecent prints, excessively revealing or tight clothing, mini-skirts, bare midriffs,
    transparent or torn pants, shorts, caps or hats inside classrooms, and slippers/clogs
    are not permitted during non-uniform days.
pipeline_tag: sentence-similarity
library_name: sentence-transformers
---

# SentenceTransformer based on sentence-transformers/all-MiniLM-L12-v2

This is a [sentence-transformers](https://www.SBERT.net) model finetuned from [sentence-transformers/all-MiniLM-L12-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L12-v2). It maps sentences & paragraphs to a 384-dimensional dense vector space and can be used for semantic textual similarity, semantic search, paraphrase mining, text classification, clustering, and more.

## Model Details

### Model Description
- **Model Type:** Sentence Transformer
- **Base model:** [sentence-transformers/all-MiniLM-L12-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L12-v2) <!-- at revision c004d8e3e901237d8fa7e9fff12774962e391ce5 -->
- **Maximum Sequence Length:** 128 tokens
- **Output Dimensionality:** 384 dimensions
- **Similarity Function:** Cosine Similarity
<!-- - **Training Dataset:** Unknown -->
<!-- - **Language:** Unknown -->
<!-- - **License:** Unknown -->

### Model Sources

- **Documentation:** [Sentence Transformers Documentation](https://sbert.net)
- **Repository:** [Sentence Transformers on GitHub](https://github.com/huggingface/sentence-transformers)
- **Hugging Face:** [Sentence Transformers on Hugging Face](https://huggingface.co/models?library=sentence-transformers)

### Full Model Architecture

```
SentenceTransformer(
  (0): Transformer({'max_seq_length': 128, 'do_lower_case': False, 'architecture': 'BertModel'})
  (1): Pooling({'word_embedding_dimension': 384, 'pooling_mode_cls_token': False, 'pooling_mode_mean_tokens': True, 'pooling_mode_max_tokens': False, 'pooling_mode_mean_sqrt_len_tokens': False, 'pooling_mode_weightedmean_tokens': False, 'pooling_mode_lasttoken': False, 'include_prompt': True})
)
```

## Usage

### Direct Usage (Sentence Transformers)

First install the Sentence Transformers library:

```bash
pip install -U sentence-transformers
```

Then you can load this model and run inference.
```python
from sentence_transformers import SentenceTransformer

# Download from the 🤗 Hub
model = SentenceTransformer("sentence_transformers_model_id")
# Run inference
sentences = [
    'Prohibited clothing on non-uniform days policy (policy)',
    'Indecent prints, excessively revealing or tight clothing, mini-skirts, bare midriffs, transparent or torn pants, shorts, caps or hats inside classrooms, and slippers/clogs are not permitted during non-uniform days.',
    'Final grades are computed using the progressive formulas in the manual: Period Grade (PG), Midterm Grade (MG), and Final Grade (FG) which combine class standing and examination components according to prescribed weights.',
]
embeddings = model.encode(sentences)
print(embeddings.shape)
# [3, 384]

# Get the similarity scores for the embeddings
similarities = model.similarity(embeddings, embeddings)
print(similarities)
# tensor([[1.0000, 0.9320, 0.0963],
#         [0.9320, 1.0000, 0.0644],
#         [0.0963, 0.0644, 1.0000]])
```

<!--
### Direct Usage (Transformers)

<details><summary>Click to see the direct usage in Transformers</summary>

</details>
-->

<!--
### Downstream Usage (Sentence Transformers)

You can finetune this model on your own dataset.

<details><summary>Click to expand</summary>

</details>
-->

<!--
### Out-of-Scope Use

*List how the model may foreseeably be misused and address what users ought not to do with the model.*
-->

<!--
## Bias, Risks and Limitations

*What are the known or foreseeable issues stemming from this model? You could also flag here known failure cases or weaknesses of the model.*
-->

<!--
### Recommendations

*What are recommendations with respect to the foreseeable issues? For example, filtering explicit content.*
-->

## Training Details

### Training Dataset

#### Unnamed Dataset

* Size: 20,000 training samples
* Columns: <code>sentence_0</code> and <code>sentence_1</code>
* Approximate statistics based on the first 1000 samples:
  |         | sentence_0                                                                        | sentence_1                                                                         |
  |:--------|:----------------------------------------------------------------------------------|:-----------------------------------------------------------------------------------|
  | type    | string                                                                            | string                                                                             |
  | details | <ul><li>min: 5 tokens</li><li>mean: 10.57 tokens</li><li>max: 23 tokens</li></ul> | <ul><li>min: 29 tokens</li><li>mean: 38.92 tokens</li><li>max: 53 tokens</li></ul> |
* Samples:
  | sentence_0                                                      | sentence_1                                                                                                                                                                                                                                                           |
  |:----------------------------------------------------------------|:---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
  | <code>Official rule on Appeals and grievance procedures</code>  | <code>Students may file appeals or grievances following the process specified in the Manual. Timelines, required forms, and steps for escalation are provided in the relevant section.</code>                                                                        |
  | <code>Explain the No Credit (NC) policy.</code>                 | <code>An NC is permanent and assigned when a student fails to take required examinations or submit essential academic requirements; it carries no credit and cannot be converted to a passing grade.</code>                                                          |
  | <code>Official rule on Grooming and appearance standards</code> | <code>Students must maintain neat and conservative grooming. Unnatural loud hair coloring, multiple visible earrings beyond one per ear, facial tattoos, and excessive piercings are disallowed. Violations may be referred to the Office of Student Affairs.</code> |
* Loss: [<code>MultipleNegativesRankingLoss</code>](https://sbert.net/docs/package_reference/sentence_transformer/losses.html#multiplenegativesrankingloss) with these parameters:
  ```json
  {
      "scale": 20.0,
      "similarity_fct": "cos_sim",
      "gather_across_devices": false
  }
  ```

### Training Hyperparameters
#### Non-Default Hyperparameters

- `per_device_train_batch_size`: 32
- `per_device_eval_batch_size`: 32
- `multi_dataset_batch_sampler`: round_robin

#### All Hyperparameters
<details><summary>Click to expand</summary>

- `overwrite_output_dir`: False
- `do_predict`: False
- `eval_strategy`: no
- `prediction_loss_only`: True
- `per_device_train_batch_size`: 32
- `per_device_eval_batch_size`: 32
- `per_gpu_train_batch_size`: None
- `per_gpu_eval_batch_size`: None
- `gradient_accumulation_steps`: 1
- `eval_accumulation_steps`: None
- `torch_empty_cache_steps`: None
- `learning_rate`: 5e-05
- `weight_decay`: 0.0
- `adam_beta1`: 0.9
- `adam_beta2`: 0.999
- `adam_epsilon`: 1e-08
- `max_grad_norm`: 1
- `num_train_epochs`: 3
- `max_steps`: -1
- `lr_scheduler_type`: linear
- `lr_scheduler_kwargs`: {}
- `warmup_ratio`: 0.0
- `warmup_steps`: 0
- `log_level`: passive
- `log_level_replica`: warning
- `log_on_each_node`: True
- `logging_nan_inf_filter`: True
- `save_safetensors`: True
- `save_on_each_node`: False
- `save_only_model`: False
- `restore_callback_states_from_checkpoint`: False
- `no_cuda`: False
- `use_cpu`: False
- `use_mps_device`: False
- `seed`: 42
- `data_seed`: None
- `jit_mode_eval`: False
- `bf16`: False
- `fp16`: False
- `fp16_opt_level`: O1
- `half_precision_backend`: auto
- `bf16_full_eval`: False
- `fp16_full_eval`: False
- `tf32`: None
- `local_rank`: 0
- `ddp_backend`: None
- `tpu_num_cores`: None
- `tpu_metrics_debug`: False
- `debug`: []
- `dataloader_drop_last`: False
- `dataloader_num_workers`: 0
- `dataloader_prefetch_factor`: None
- `past_index`: -1
- `disable_tqdm`: False
- `remove_unused_columns`: True
- `label_names`: None
- `load_best_model_at_end`: False
- `ignore_data_skip`: False
- `fsdp`: []
- `fsdp_min_num_params`: 0
- `fsdp_config`: {'min_num_params': 0, 'xla': False, 'xla_fsdp_v2': False, 'xla_fsdp_grad_ckpt': False}
- `fsdp_transformer_layer_cls_to_wrap`: None
- `accelerator_config`: {'split_batches': False, 'dispatch_batches': None, 'even_batches': True, 'use_seedable_sampler': True, 'non_blocking': False, 'gradient_accumulation_kwargs': None}
- `parallelism_config`: None
- `deepspeed`: None
- `label_smoothing_factor`: 0.0
- `optim`: adamw_torch_fused
- `optim_args`: None
- `adafactor`: False
- `group_by_length`: False
- `length_column_name`: length
- `project`: huggingface
- `trackio_space_id`: trackio
- `ddp_find_unused_parameters`: None
- `ddp_bucket_cap_mb`: None
- `ddp_broadcast_buffers`: False
- `dataloader_pin_memory`: True
- `dataloader_persistent_workers`: False
- `skip_memory_metrics`: True
- `use_legacy_prediction_loop`: False
- `push_to_hub`: False
- `resume_from_checkpoint`: None
- `hub_model_id`: None
- `hub_strategy`: every_save
- `hub_private_repo`: None
- `hub_always_push`: False
- `hub_revision`: None
- `gradient_checkpointing`: False
- `gradient_checkpointing_kwargs`: None
- `include_inputs_for_metrics`: False
- `include_for_metrics`: []
- `eval_do_concat_batches`: True
- `fp16_backend`: auto
- `push_to_hub_model_id`: None
- `push_to_hub_organization`: None
- `mp_parameters`: 
- `auto_find_batch_size`: False
- `full_determinism`: False
- `torchdynamo`: None
- `ray_scope`: last
- `ddp_timeout`: 1800
- `torch_compile`: False
- `torch_compile_backend`: None
- `torch_compile_mode`: None
- `include_tokens_per_second`: False
- `include_num_input_tokens_seen`: no
- `neftune_noise_alpha`: None
- `optim_target_modules`: None
- `batch_eval_metrics`: False
- `eval_on_start`: False
- `use_liger_kernel`: False
- `liger_kernel_config`: None
- `eval_use_gather_object`: False
- `average_tokens_across_devices`: True
- `prompts`: None
- `batch_sampler`: batch_sampler
- `multi_dataset_batch_sampler`: round_robin
- `router_mapping`: {}
- `learning_rate_mapping`: {}

</details>

### Training Logs
| Epoch | Step | Training Loss |
|:-----:|:----:|:-------------:|
| 0.8   | 500  | 0.6267        |
| 1.6   | 1000 | 0.614         |
| 2.4   | 1500 | 0.6082        |


### Framework Versions
- Python: 3.12.12
- Sentence Transformers: 5.1.2
- Transformers: 4.57.1
- PyTorch: 2.8.0+cu126
- Accelerate: 1.11.0
- Datasets: 4.0.0
- Tokenizers: 0.22.1

## Citation

### BibTeX

#### Sentence Transformers
```bibtex
@inproceedings{reimers-2019-sentence-bert,
    title = "Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks",
    author = "Reimers, Nils and Gurevych, Iryna",
    booktitle = "Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing",
    month = "11",
    year = "2019",
    publisher = "Association for Computational Linguistics",
    url = "https://arxiv.org/abs/1908.10084",
}
```

#### MultipleNegativesRankingLoss
```bibtex
@misc{henderson2017efficient,
    title={Efficient Natural Language Response Suggestion for Smart Reply},
    author={Matthew Henderson and Rami Al-Rfou and Brian Strope and Yun-hsuan Sung and Laszlo Lukacs and Ruiqi Guo and Sanjiv Kumar and Balint Miklos and Ray Kurzweil},
    year={2017},
    eprint={1705.00652},
    archivePrefix={arXiv},
    primaryClass={cs.CL}
}
```

<!--
## Glossary

*Clearly define terms in order to be accessible across audiences.*
-->

<!--
## Model Card Authors

*Lists the people who create the model card, providing recognition and accountability for the detailed work that goes into its construction.*
-->

<!--
## Model Card Contact

*Provides a way for people who have updates to the Model Card, suggestions, or questions, to contact the Model Card authors.*
-->