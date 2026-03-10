import os
import json
import math
import time
import asyncio
from typing import Union, Literal, Optional, Iterator, List, Any, Dict
from tqdm import tqdm
import copy
from pathlib import Path
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.stdout.reconfigure(encoding='utf-8')

from AgentInit.utils.const import AgentInit_ROOT
from experiments_Autogen.accuracy import Accuracy
from AgentInit.utils.globals import Cost, PromptTokens, CompletionTokens, Time
import re
from tenacity import retry, wait_random_exponential, stop_after_attempt, wait_fixed
from experiments_Autogen.Autogen import run_team_chat
from experiments_Autogen.agentinit.manager import Manager


async def evaluate(
        dataset,
        limit_questions: Optional[int] = None,
        eval_batch_size: int = 4,
        args=None,
) -> float:
    accuracy = Accuracy()

    current_time = Time.instance().value or time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
    result_dir = Path(f"{AgentInit_ROOT}/result/mmlu")
    result_dir.mkdir(parents=True, exist_ok=True)

    # 🌟 核心新增 1：定义检查点（Checkpoint）文件路径
    # 使用模型名字来区分文件，防止不同模型的结果混淆
    safe_model_name = args.llm_name.replace("/", "_")
    checkpoint_file = result_dir / f"checkpoint_{safe_model_name}_{args.domain}.jsonl"

    processed_count = 0
    # 🌟 核心新增 2：启动时读取检查点，恢复进度
    if checkpoint_file.exists():
        with open(checkpoint_file, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    item = json.loads(line)
                    # 恢复 Accuracy 统计
                    accuracy.update(item["Postprocessed_Answer"], item["Answer"])
                    processed_count += 1
        print(f"📦 发现断点记录！已恢复 {processed_count} 条数据的进度，当前准确率：")
        accuracy.print()

    # 🌟 核心修改 3：在 Loader 中跳过已经跑过的数据
    def eval_loader(batch_size: int) -> Iterator[List[Any]]:
        records = []
        for i_record, record in enumerate(dataset):
            if limit_questions is not None:
                if i_record >= limit_questions:
                    break

            # 跳过已经处理过的记录
            if i_record < processed_count:
                continue

            records.append(record)
            if len(records) >= batch_size:
                yield records
                records = []
        if len(records) > 0:
            yield records
        return

    data_len = min(len(dataset), limit_questions) if limit_questions is not None else len(dataset)

    # 如果已经全部跑完，直接返回
    if processed_count >= data_len:
        print("✅ 所有数据已评估完毕！")
        return accuracy.get()

    @retry(wait=wait_fixed(10), stop=stop_after_attempt(5))
    async def process_record(record, llm_name, mode):
        input_dict = dataset.record_to_input(record)
        question = input_dict["task"]
        manager = Manager(llm_name=llm_name)
        roles = await manager._act(question)
        response = await run_team_chat(question, llm_name, roles, args.domain)
        return response

    # 🌟 核心修改 4：初始化进度条时，加入 initial 参数，使得进度条从断点处开始
    pbar = tqdm(total=data_len, initial=processed_count, desc="Evaluating", dynamic_ncols=True)

    async def process_and_update(record, llm_name, mode):
        try:
            res = await process_record(record, llm_name, mode)
            return res
        finally:
            pbar.update(1)

    for i_batch, record_batch in enumerate(eval_loader(batch_size=eval_batch_size)):
        tqdm.write(80 * '-')
        start_ts = time.time()
        answer_log_probs = []

        for record in record_batch:
            answer_log_probs.append(asyncio.create_task(process_and_update(record, args.llm_name, args.mode)))
        raw_answers = await asyncio.gather(*answer_log_probs)

        tqdm.write(f"Batch time {time.time() - start_ts:.3f}")
        for raw_answer, record in zip(raw_answers, record_batch):
            cleaned_answer = re.sub(r'<think>.*?</think>', '', raw_answer, flags=re.DOTALL).strip()
            answer = dataset.postprocess_answer(cleaned_answer)
            correct_answer = dataset.record_to_target_answer(record)

            accuracy.update(answer, correct_answer)
            accuracy.print()

            # 🌟 核心新增 5：将提取后的 answer 也存进去，方便下次启动时恢复 Accuracy
            updated_item = {
                "Question": dataset.record_to_input(record)['task'],
                "Answer": correct_answer,
                "Postprocessed_Answer": answer,
                "Response": raw_answer,
            }
            # tqdm.write(str(updated_item))

            # 🌟 核心新增 6：将单条结果实时追加写入（Append）到检查点文件
            with open(checkpoint_file, 'a', encoding='utf-8') as f:
                f.write(json.dumps(updated_item, ensure_ascii=False) + '\n')

        tqdm.write(f"Cost {Cost.instance().value}")
        tqdm.write(f"PromptTokens {PromptTokens.instance().value}")
        tqdm.write(f"CompletionTokens {CompletionTokens.instance().value}")

    pbar.close()
    accuracy.print()
    print("Done!")

    return accuracy.get()