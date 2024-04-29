# KoInFoBench

KoInFoBench is a specialized evaluation dataset designed to assess the performance of Large Language Models (LLMs) on capabilities of Korean instructions following.

Inspired by [InFoBench](https://huggingface.co/datasets/kqsong/InFoBench) dataset, we extends their concpet by focusing on the nuances and features of Korean language.

This reporsitory is to provide the guidances 
  1. for generating responses from Proprietary LLM APIs
  2. for evaluating whether the respones are well-generated as following instructions

📄 Paper is under writing and open soon! <br>
🤗 We welcome any opinions and contributions for our dataset [KoInFoBench](https://huggingface.co/datasets/kifai/KoInFoBench).

## Eavluation with KoInFoBench

### Setting Environments 

Download this github repository
```shell
git clone https://github.com/KIFAI/KoInFoBench.git
cd KoInFoBench
```

Create conda environment for this project and install python packages
```shell
conda create -n koinfobench python=3.10
conda activate koinfobench

pip install -r requirements.txt
```
Copy the file `.env.example` to `.env` and replace the values with your own API keys

```shell
OPENAI_API_KEY="YOUR OWN API KEY"
CLAUDE_API_KEY="YOUR OWN API KEY"
GEMINI_API_KEY="YOUR OWN API KEY"
HYPERCLOVAX_API_KEY="YOUR OWN API KEY"
HYPERCLOVAX_API_KEY_PRIMARY_EVAL="YOUR OWN API KEY"
```

### Step1. Check dataset usage

You can directly download it with huggingface datasets.

from datasets import load_dataset

dataset = load_dataset("kifai/KoInFoBench")


    
### 2. Generating responses of instructions

To generate responses from
- A specfic model
    ```shell
    python generation.py \
    --api_type openai  \
    --models gpt-4-0125-preview
    ```
- Multiple models (just append `--models` and `model_name`)
    ```shell
    python generation.py \
    --api_type openai \
    --models gpt-3.5-turbo-0125 \
    --models gpt-4-0125-preview \
    --models gpt-4-turbo-2024-04-09
    ```
- Multiple models of all api types
    ```shell
    python generation.py \
    --api_type all \
    --models gpt-3.5-turbo-0125 \
    --models gpt-4-0125-preview \
    --models claude-3-opus-20240229 \
    --models gemini-1.5-pro \
    --models hpx003
    ```

Generated responses are saved to the output_dir as `jsonl` file (e.g. `generation_{MODEL_NAME}.jsonl`). We share the generated responses using our own API budget in the directory `generations`. You can compare the result of various LLMs how they generate responses as Korean instructions.
```json
{"output": "Here is a comparison table of similar products based on price, rating, format, and paper size:\n\n| **Product** | **Price** | **Rating** | **Format** | **Paper Size** |\n| --- | --- | --- | --- | --- |\n| 모닝글로리 3000 테마컬러 원링 노트 4권세트 PP 스프링노트 16절 줄노트 대학 모닝펜 | ₩25,000 | 4.5/5 | Ring-bound | A5 (148 x 210 mm) |\n| Morning Glory 2000 테마컬러 원링 노트 3권세트 PP 스프링노트 16절 줄노트 대학 모닝펜 | ₩20,000 | 4.4/5 | Ring-bound | A5 (148 x 210 mm) |\n| Rhodia Webnotebook A5 96 sheets 80gsm | ₩30,000 | 4.7/5 | Sewn-bound | A5 (148 x 210 mm) |\n| Quo Vadis Habana A5 128 pages 85gsm | ₩35,000 | 4.6/5 | Sewn-bound | A5 (148 x 210 mm) |\n| Tsubame 3000 원링 노트 4권세트 PP 스프링노트 16절 줄노트 대학 | ₩22,000 | 4.3/5 | Ring-bound | A5 (148 x 210 mm) |\n\nNote: Prices may vary depending on the retailer and availability. Ratings are based on average customer reviews.", "input": "", "id": "1", "model_name": "meta-llama/Meta-Llama-3-70B-Instruct"}
{"output": "Here are the extracted option information from the product name:\n\n* Brand: MSI\n* Model: 지포스 RTX 4070 SUPER\n* Series: 게이밍 X\n* Form Factor: 슬림\n* Memory Type: D6X\n* Capacity: 12GB\n* Cooler: 트라이프로져3\n\nLet me know if you need any further assistance! 😊", "input": "", "id": "2", "model_name": "meta-llama/Meta-Llama-3-70B-Instruct"}
...
```

### 3. Evaluating responses automatically using GPT-4
Before running the following scripts, make sure the generated files are placed on the directory `generation` or the directory you manually specified.
`gpt-4-0613` is the default evaluation model to judge the responses using decomposed questions. A instruction comprises multiple decomposed questions to judge whether a response is well-generated as following the instruction.

```shell
python evaluation.py \
--input_dir generation \
--eval_model gpt-4-0613 \
--seed 1234

# Model                           DRFR
# -----                           ----
# claude-3-opus-20240229         0.850
# hpx003                         0.738
# gpt-3.5-turbo-0125             0.734
# gpt-4-0125-preview             0.824
# gemini-1.5-pro                 0.811
# gpt-4-turbo-2024-04-09         0.880
```

Each data will include an "answers" key in the format of List[str] which composes of "Yes" or "No" answers to each decomposed question. The final output evaluation file will be saved in JSON format at location `{OUTPUT_DIR}/result.json`.

<details>
<summary>example of result.json</summary>
<pre markdown="1" code='json'>
{
  "claude-3-opus-20240229": {
    "filepath": "./generations_claude-3-opus-20240229.jsonl",
    "drfr": 0.8497854077253219,
    "evaluations": [
      {
        "input": "상품: 모닝글로리 3000 테마컬러 원링 노트 4권세트 PP 스프링노트 16절 줄노트 대학 모닝펜",
        "output": "| 특징 | 모닝글로리 3000 테마컬러 원링 노트 4권세트 | 아이바라기 원링 PP 노트 20절 8mm 줄노트 5권 | 아르누보 제본노트 16절 줄노트 10권 |\n|:-|:-|:-|:-|\n| 가격 | 5,000원 | 6,900원 | 5,000원 |\n| 별점 | 4.5 | 4.0 | 4.5 |\n| 내지 형태 | 줄노트 | 줄노트 | 줄노트 |\n| 종이 사이즈 | 16절 | 20절 | 16절 | \n| 특징 | 4권 세트, PP 스프링 제본, 대학 모닝펜 증정 | 5권 세트, PP 스프링 제본, 8mm 줄간격 | 10권 세트, 제본노트 |\n\n위 표에서 제시된 세 가지 상품은 모두 줄노트 형태의 노트이며, 16절 또는 20절 크기입니다. 가격대도 5,000원에서 7,000원 사이로 유사합니다. \n\n모닝글로리 상품은 4권 세트로 구성되어 있고, PP 스프링 제본 방식을 사용하였으며, 대학 모닝펜이 증정품으로 제공됩니다. \n\n아이바라기 상품은 5권 세트로 구성되어 있고, 역시 PP 스프링 제본 방식을 사용하였으며, 8mm 줄간격의 노트입니다. \n\n아르누보 상품은 10권 세트로 구성되어 있고, 제본노트 형태입니다. \n\n따라서 위 세 상품은 모두 줄노트 형태에 유사한 크기와 가격대를 가지고 있어 서로 비교 가능한 상품으로 볼 수 있습니다.",
        "messages": [
          {
            "role": "user",
            "content": "Based on the provided Input (if any) and Generated Text, answer the ensuing Questions with either a YES or NO choice. Your selection should be based on your judgment as well as the following rules:\n\n- YES: Select 'YES' if the generated text entirely fulfills the condition specified in the question. However, note that even minor inaccuracies exclude the text from receiving a 'YES' rating. As an illustration. consider a question that asks. \"Does each sentence in the generated text use a second person?” If even one sentence does not use the second person, the answer should NOT be 'YES'. To qualify for a 'YES' rating, the generated text must be entirely accurate and relevant to the question\n\n- NO: Opt for 'NO' if the generated text fails to meet the question's requirements or provides no information that could be utilized to answer the question. For instance, if the question asks. \"Is the second sentence in the generated text a compound sentence?\" and the generated text only has one sentence. it offers no relevant information to answer the question. Consequently, the answer should be 'NO'.'''\n\nInput:\n\"상품: 모닝글로리 3000 테마컬러 원링 노트 4권세트 PP 스프링노트 16절 줄노트 대학 모닝펜\"\n\nGenerated Text:\n\"| 특징 | 모닝글로리 3000 테마컬러 원링 노트 4권세트 | 아이바라기 원링 PP 노트 20절 8mm 줄노트 5권 | 아르누보 제본노트 16절 줄노트 10권 |\n|:-|:-|:-|:-|\n| 가격 | 5,000원 | 6,900원 | 5,000원 |\n| 별점 | 4.5 | 4.0 | 4.5 |\n| 내지 형태 | 줄노트 | 줄노트 | 줄노트 |\n| 종이 사이즈 | 16절 | 20절 | 16절 | \n| 특징 | 4권 세트, PP 스프링 제본, 대학 모닝펜 증정 | 5권 세트, PP 스프링 제본, 8mm 줄간격 | 10권 세트, 제본노트 |\n\n위 표에서 제시된 세 가지 상품은 모두 줄노트 형태의 노트이며, 16절 또는 20절 크기입니다. 가격대도 5,000원에서 7,000원 사이로 유사합니다. \n\n모닝글로리 상품은 4권 세트로 구성되어 있고, PP 스프링 제본 방식을 사용하였으며, 대학 모닝펜이 증정품으로 제공됩니다. \n\n아이바라기 상품은 5권 세트로 구성되어 있고, 역시 PP 스프링 제본 방식을 사용하였으며, 8mm 줄간격의 노트입니다. \n\n아르누보 상품은 10권 세트로 구성되어 있고, 제본노트 형태입니다. \n\n따라서 위 세 상품은 모두 줄노트 형태에 유사한 크기와 가격대를 가지고 있어 서로 비교 가능한 상품으로 볼 수 있습니다.\"\n\nQuestion:\n생성된 텍스트에는 표가 있습니까?\n"
          },
          {
            "role": "assistant",
            "content": "Yes\n"
          },
          {
            "role": "user",
            "content": "Question:\n입력으로 제공된 상품 항목이 표 안에 포함되어 있습니까?\n"
          },
          {
            "role": "assistant",
            "content": "Yes\n"
          },
          {
            "role": "user",
            "content": "Question:\n생성된 표 안의 모든 상품들은 입력으로 주어진 상품과 같은 유형의 제품입니까?\n"
          },
          {
            "role": "assistant",
            "content": "Yes\n"
          },
          {
            "role": "user",
            "content": "Question:\n표 안에 작성된 상품 항목의 특징들이 가격, 별점, 내지 형태, 종이 사이즈 각 컬럼에 알맞는 내용으로 작성되었습니까?\n"
          },
          {
            "role": "assistant",
            "content": "No\n"
          }
        ],
        "answers": [
          "Yes",
          "Yes",
          "Yes",
          "No"
        ]
      },
      ...
    ]
  }
}
```
</pre>
</details>
<br>
Please note that the results of an automated evaluation using GPT-4 may differ from the results of an in-person assessment by human experts. In our setting, the degree of alignment between human experts and llm judgement is about **83%**.

## Evaluation Result (ours)

### DRFR
Decomposed Requirements Following Ratio(DRFR) is the metric to evaluate how LLMs accurately respond to the instruction/input.
This metric calculates the average accuracy across answers to the decomposed questions for each instruction.
The following is the summary of the model performance on our dataset.

| Model                                       | H_DRFR     | A_DRFR | Alignment |
|---------------------------------------------|------------|--------|-----------|
| **claude-3-opus-20240229**                  | **0.854**  | 0.850  | 87%       |
| **gpt-4-turbo-2024-04-09**                  | 0.850      | 0.880  | 87%       |
| **gpt-4-0125-preview**                      | 0.824      | 0.824  | 83%       |
| **gemini-1.5-pro**                          | 0.773      | 0.811  | 83%       |
| **meta-llama/Meta-Llama-3-70B-Instruct**   | 0.747      | 0.863  | 84%       |
| **hpx003**                                  | 0.691      | 0.738  | 83%       |
| **gpt-3.5-turbo-0125**                      | 0.678      | 0.734  | 82%       |
| **yanolja/EEVE-Korean-Instruct-10.8B-v1.0** | 0.597      | 0.730  | 79%       |

- `H_DRFR`: The accuracy of model responses as evaluated by the human expert
- `A_DRFR`: The accuracy of model responses automatically evaluated by GPT-4 as employing the capability of LLM-as-a-judge
- `Alignment`: The degree of agreement or consistency between the human and automated evaluation

## Additional Information

### License Information

This dataset is released under the [MIT LISENCE](https://github.com/KIFAI/KoInfoBench/blob/main/LICENSE)

### Citation
```
@article{,
      title={KoInFoBench: Evaluations for Korean Insturction-Following Capabilities of LLM}, 
      author={Sungwoo Oh, Sungjun Kown, Donggyu Kim},
      year={2024},
      eprint={},
}
```
