# Báo cáo cấu hình huấn luyện

Mô hình dùng là **Llama 3.2 3B**. Cả hai bước huấn luyện (SFT và GRPO) đều dùng **LoRA** (chỉ chỉnh một phần tham số của mô hình để tiết kiệm bộ nhớ và thời gian). SFT chạy bằng LLaMA-Factory với file cấu hình YAML; GRPO chạy bằng MS-SWIFT qua script `scripts/train.sh`. Dưới đây liệt kê toàn bộ tham số cấu hình cho từng bước.

---

## 1. Cấu hình SFT (Supervised Fine-Tuning)

SFT dùng để dạy mô hình bắt chước cách trả lời có lời giải (reasoning) theo từng bộ dữ liệu: **origin** (gốc), **low** (lời giải ngắn), **medium** (trung bình), **high** (lời giải dài). File cấu hình nằm trong `LLaMA-Factory/examples/train_lora/` (ví dụ `llama_1mode.yaml`, `llama_3mode.yaml`). Chạy lệnh train từ thư mục LLaMA-Factory.

### 1.1. Mô hình và cách train


| Tham số            | Giá trị                          | Ý nghĩa                                          |
| ------------------ | -------------------------------- | ------------------------------------------------ |
| model_name_or_path | meta-llama/Llama-3.2-3B-Instruct | Đường dẫn hoặc tên mô hình gốc trên Hugging Face |
| stage              | sft                              | Giai đoạn SFT                                    |
| do_train           | true                             | Bật chế độ train                                 |
| finetuning_type    | lora                             | Dùng LoRA, không train full mô hình              |
| template           | llama3                           | Cách format hội thoại cho Llama 3                |


### 1.2. LoRA


| Tham số      | Giá trị | Ý nghĩa                                                         |
| ------------ | ------- | --------------------------------------------------------------- |
| lora_rank    | 32      | Hạng của ma trận LoRA (số càng lớn dung lượng chỉnh càng nhiều) |
| lora_alpha   | 64      | Hệ số scale cho LoRA                                            |
| lora_target  | all     | Áp LoRA lên toàn bộ layer được hỗ trợ                           |
| lora_dropout | 0.05    | Tỷ lệ dropout trong LoRA để tránh overfit                       |


### 1.3. Dữ liệu


| Tham số                   | Giá trị                                                       | Ý nghĩa                                                                                                  |
| ------------------------- | ------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------- |
| dataset                   | sft_medium (1 mode) hoặc sft_low,sft_medium,sft_high (3 mode) | Tên dataset train: một trong low / medium / high hoặc cả ba; origin cần đăng ký riêng trong dataset_info |
| eval_dataset              | sft_grpo_val_low,sft_grpo_val_medium,sft_grpo_val_high        | Dataset đánh giá (giữ nguyên khi train 1 mode hoặc 3 mode)                                               |
| packing                   | true                                                          | Gộp nhiều mẫu vào một sequence để tận dụng độ dài tối đa                                                 |
| cutoff_len                | 3072                                                          | Độ dài tối đa một sequence (câu hỏi + câu trả lời) tính theo token                                       |
| overwrite_cache           | true                                                          | Ghi đè cache tiền xử lý dữ liệu                                                                          |
| preprocessing_num_workers | 16                                                            | Số process tiền xử lý                                                                                    |
| dataloader_num_workers    | 4                                                             | Số worker load batch khi train                                                                           |


### 1.4. Huấn luyện (learning rate, batch, epoch)


| Tham số                     | Giá trị                   | Ý nghĩa                                                                 |
| --------------------------- | ------------------------- | ----------------------------------------------------------------------- |
| learning_rate               | 2.0e-4                    | Tốc độ học                                                              |
| per_device_train_batch_size | 16                        | Số mẫu mỗi batch trên một GPU                                           |
| gradient_accumulation_steps | 2                         | Cộng dồn gradient 2 bước rồi mới cập nhật → batch hiệu dụng = 16×2 = 32 |
| num_train_epochs            | 1.0 (hoặc 0.3 cho phase1) | Số epoch đi qua toàn bộ data                                            |
| lr_scheduler_type           | cosine                    | Giảm learning rate theo dạng cosine                                     |
| warmup_ratio                | 0.1                       | 10% số bước đầu tăng dần learning rate                                  |
| seed                        | 42                        | Hạt ngẫu nhiên để lặp lại thí nghiệm                                    |
| bf16                        | true                      | Dùng số dấu chấm động 16 bit để tiết kiệm bộ nhớ và tăng tốc            |
| ddp_timeout                 | 180000000                 | Timeout khi train đa GPU                                                |
| resume_from_checkpoint      | null                      | Không tiếp tục từ checkpoint cũ (có thể đổi thành đường dẫn nếu resume) |


### 1.5. Lưu mô hình và log


| Tham số              | Giá trị                                                | Ý nghĩa                                                                         |
| -------------------- | ------------------------------------------------------ | ------------------------------------------------------------------------------- |
| output_dir           | saves/llama_3b_sft_1mode_final (hoặc tên khác tùy run) | Thư mục lưu checkpoint                                                          |
| run_name             | llama_3b_sft_1mode_final (hoặc tên khác)               | Tên run trên wandb/tensorboard                                                  |
| logging_steps        | 5                                                      | Bao nhiêu bước thì ghi log một lần                                              |
| save_steps           | 20 (hoặc 40 với 3 mode)                                | Bao nhiêu bước thì lưu checkpoint một lần                                       |
| save_total_limit     | 2                                                      | Chỉ giữ tối đa 2 checkpoint, xóa bản cũ hơn                                     |
| plot_loss            | true                                                   | Vẽ đồ thị loss                                                                  |
| overwrite_output_dir | false                                                  | Không ghi đè thư mục output nếu đã có                                           |
| save_only_model      | false                                                  | Lưu cả optimizer/state ngoài weights (để resume)                                |
| report_to            | wandb                                                  | Gửi log lên Weights & Biases (có thể đổi none / tensorboard / swanlab / mlflow) |


### 1.6. Đánh giá trong lúc train


| Tham số                    | Giá trị                       | Ý nghĩa                                            |
| -------------------------- | ----------------------------- | -------------------------------------------------- |
| eval_strategy              | steps                         | Đánh giá theo số bước                              |
| eval_steps                 | 20 (hoặc 40 với 3 mode)       | Bao nhiêu bước thì chạy eval một lần               |
| per_device_eval_batch_size | 1 (hoặc 2)                    | Batch size khi đánh giá                            |
| metric_for_best_model      | eval_sft_grpo_val_medium_loss | Dùng loss trên val medium để chọn mô hình tốt nhất |
| eval_on_each_dataset       | true                          | Báo loss trên từng dataset val (low, medium, high) |


### 1.7. Tăng tốc / phần cứng


| Tham số             | Giá trị | Ý nghĩa                    |
| ------------------- | ------- | -------------------------- |
| flash_attn          | fa2     | Dùng Flash Attention 2     |
| enable_liger_kernel | true    | Bật kernel tối ưu Liger    |
| use_unsloth_gc      | true    | Tối ưu bộ nhớ kiểu Unsloth |


---

## 2. Cấu hình GRPO (Group Relative Policy Optimization)

GRPO là bước tinh chỉnh bằng reinforcement learning: mô hình sinh nhiều câu trả lời, được chấm điểm (reward) theo đúng đáp án và độ ngắn gọn, rồi cập nhật để tăng điểm. Chạy từ thư mục gốc repo bằng `bash scripts/train.sh`. Có thể đổi bộ dữ liệu (origin / low / med / high) bằng cách đổi biến môi trường **MODEL** (checkpoint SFT tương ứng), **DATASET** (file JSONL GRPO), **OUTPUT_DIR** (thư mục lưu kết quả).

### 2.1. Mô hình và dữ liệu (qua biến môi trường)


| Tham số              | Giá trị mặc định                              | Ý nghĩa                                                                          |
| -------------------- | --------------------------------------------- | -------------------------------------------------------------------------------- |
| MODEL                | KoiiVN/final_llama_3b_sft_origin              | Mô hình hoặc checkpoint SFT dùng làm điểm bắt đầu (Hub hoặc đường dẫn local)     |
| MODEL_TYPE           | llama3_2                                      | Loại kiến trúc (Llama 3.2)                                                       |
| DATASET              | merged_grpo_data.jsonl (đường dẫn trong repo) | File JSONL chứa câu hỏi và đáp án đúng (có trường query, label dạng \boxed{...}) |
| OUTPUT_DIR           | outputs/llama_origin_grpo                     | Thư mục lưu checkpoint GRPO                                                      |
| CUDA_VISIBLE_DEVICES | 2                                             | GPU nào dùng để train (có thể đổi 0, 1, 0,1...)                                  |


### 2.2. Cách train và LoRA


| Tham số                     | Giá trị | Ý nghĩa                                                                                          |
| --------------------------- | ------- | ------------------------------------------------------------------------------------------------ |
| rlhf_type                   | grpo    | Dùng thuật toán GRPO                                                                             |
| train_type                  | lora    | Chỉ train LoRA                                                                                   |
| loss_type                   | dapo    | Biến thể DAPO của GRPO                                                                           |
| max_length                  | 3072    | Độ dài tối đa một sequence (prompt + phần model sinh); đồng thời giới hạn luôn độ dài generation |
| per_device_train_batch_size | 8       | Số mẫu mỗi batch trên một GPU                                                                    |
| gradient_accumulation_steps | 2       | Cộng dồn gradient 2 bước → batch hiệu dụng 8×2 = 16                                              |
| max_steps                   | 300     | Train 300 bước rồi dừng                                                                          |


*Các tham số không ghi trong script, dùng mặc định của MS-SWIFT:*


| Tham số       | Giá trị mặc định | Ý nghĩa                                                              |
| ------------- | ---------------- | -------------------------------------------------------------------- |
| learning_rate | 1e-4             | Tốc độ học (LoRA)                                                    |
| lora_rank     | 8                | Hạng LoRA (nếu muốn giống SFT có thể thêm --lora_rank 32 vào script) |
| lora_alpha    | 32               | Scale LoRA (có thể thêm --lora_alpha 32 hoặc 64)                     |


### 2.3. Reward (điểm thưởng)


| Tham số        | Giá trị       | Ý nghĩa                                                                                                                                               |
| -------------- | ------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------- |
| reward_funcs   | grpo_accuracy | Hàm reward chính: so đáp án sinh ra với đáp án đúng (\boxed{...}), đúng thì điểm cao; cộng thêm bonus nếu câu trả lời không lặp từ (repetition bonus) |
| reward_weights | 1             | Trọng số cho từng hàm reward (hiện chỉ 1 hàm nên là 1)                                                                                                |


*Các hàm reward khác có thể thêm (khi dùng nhiều hàm thì cần thêm reward_weights tương ứng):*


| Tên           | Ý nghĩa                                        |
| ------------- | ---------------------------------------------- |
| grpo_accuracy | Đúng đáp án + bonus ngắn gọn (đang dùng)       |
| accuracy      | Chỉ tính đúng/sai đáp án                       |
| format        | Thưởng nếu đúng format ... và ...              |
| react_format  | Thưởng format ReAct (Action, Action Input)     |
| repetition    | Phạt lặp n-gram (câu lặp từ bị trừ điểm)       |
| cosine        | Kết hợp đúng đáp án và độ dài theo kiểu cosine |
| soft_overlong | Phạt nếu sinh quá dài so với ngưỡng            |


Ví dụ dùng hai reward: thêm `format` với trọng số 0.2 → `--reward_funcs grpo_accuracy format --reward_weights 1 0.2`.

### 2.4. Sinh câu khi train (rollout)


| Tham số         | Giá trị | Ý nghĩa                                                    |
| --------------- | ------- | ---------------------------------------------------------- |
| num_generations | 8       | Mỗi câu hỏi sinh 8 câu trả lời để tính reward và advantage |
| temperature     | 1.0     | Độ ngẫu nhiên khi sinh (1.0 là khá đa dạng)                |


### 2.5. Lưu và log


| Tham số          | Giá trị | Ý nghĩa                                   |
| ---------------- | ------- | ----------------------------------------- |
| save_steps       | 100     | Bao nhiêu bước thì lưu checkpoint một lần |
| save_total_limit | 4       | Chỉ giữ tối đa 4 checkpoint               |
| logging_steps    | 10      | Bao nhiêu bước thì in log một lần         |


### 2.6. Khác


| Tham số                | Giá trị      | Ý nghĩa                                                     |
| ---------------------- | ------------ | ----------------------------------------------------------- |
| bf16                   | true         | Dùng số dấu chấm động 16 bit                                |
| gradient_checkpointing | true         | Tiết kiệm bộ nhớ bằng cách tính lại một phần trong backward |
| warmup_ratio           | 0.05         | 5% số bước đầu tăng dần learning rate                       |
| use_hf                 | 1 (mặc định) | Dùng Hugging Face; đặt USE_HF=0 trong env nếu không cần     |


---

## Tóm tắt nhanh

- **SFT:** Độ dài tối đa (cutoff_len) 3072 token, LoRA rank 32, learning rate 2e-4. Dataset: một hoặc nhiều trong sft_low, sft_medium, sft_high (và origin nếu đã cấu hình). Mọi tham số trên đều có trong file YAML (ví dụ `llama_1mode.yaml`, `llama_3mode.yaml`).
- **GRPO:** Độ dài tối đa (max_length) 3072, LoRA rank mặc định 8 (có thể chỉnh 32), learning rate mặc định 1e-4. Reward chính: grpo_accuracy; có thể thêm format, repetition và reward_weights tương ứng. Phần lớn tham số nằm trong `scripts/train.sh`; đổi MODEL, DATASET, OUTPUT_DIR để train theo từng bộ (origin / low / med / high).

