import sys
import os
import argparse
import yaml
import json
import time
import asyncio
from pathlib import Path
import torch
import torch.nn.functional as F
import copy
from typing import List,Union,Literal
import random
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.stdout.reconfigure(encoding='utf-8')
from AgentInit.llm.llm_registry import LLMRegistry
from AgentInit.utils.const import AgentInit_ROOT
from AgentInit.graph.graph import Graph
from AgentInit.tools.reader.readers import JSONReader, JSONLReader
from AgentInit.utils.globals import Time
from AgentInit.utils.globals import Cost, PromptTokens, CompletionTokens
from AgentInit.utils.utils import nuclear_norm,frobenius_norm
from datasets.gsm8k_dataset import svamp_data_process,gsm_get_predict, gsm_data_process,multiarith_data_process
from datasets.aqua_dataset import aqua_data_process,aqua_get_predict
from AgentInit.utils.globals import PromptTokens, CompletionTokens
from AgentInit.agents.agent_registry import AgentRegistry
from tenacity import retry, wait_random_exponential, stop_after_attempt, wait_fixed
from experiments_Autogen.Autogen import run_team_chat
import re
from experiments_Autogen.agentinit.manager import Manager
import collections
def load_result(result_file):
    if not result_file.exists():
        with open(result_file, 'w',encoding='utf-8') as file:
            json.dump([], file)

    with open(result_file, 'r',encoding='utf-8') as file:
        data = json.load(file)
    return data

def dataloader(data_list, batch_size, i_batch):
    return data_list[i_batch*batch_size:i_batch*batch_size + batch_size]

def load_config(config_path):
    with open(config_path, 'r',encoding='utf-8') as file:
        return yaml.safe_load(file)
    
def parse_args():
    parser = argparse.ArgumentParser(description="Experiments on gsm8k")
    parser.add_argument("--dataset_json", type=str, default="datasets/MultiArith/test.json")
    parser.add_argument("--result_file", type=str, default=None)
    parser.add_argument("--llm_name", type=str, default="gpt-3.5-turbo")
    parser.add_argument('--mode', type=str, default=None)
    parser.add_argument('--domain',type=str,default='gsm8k')
    parser.add_argument('--batch_size',type = int, default = 20)
    args = parser.parse_args()
    result_path = AgentInit_ROOT / "result"
    os.makedirs(result_path, exist_ok=True)
    return args

async def main():
    args = parse_args()
    result_file = None
    dataset = JSONReader.parse_file(args.dataset_json)
    dataset = multiarith_data_process(dataset)
    train_dataset = JSONReader.parse_file('datasets/MultiArith/train.json')
    train_dataset = multiarith_data_process(train_dataset)

    current_time = Time.instance().value or time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
    Time.instance().value = current_time
    result_dir = Path(f"{AgentInit_ROOT}/result/MultiArith")
    result_dir.mkdir(parents=True, exist_ok=True)
    total_solved, total_executed = (0, 0)

    # 断点：定义 checkpoint 文件路径
    safe_model_name = args.llm_name.replace("/", "_")
    checkpoint_file = result_dir / f"checkpoint_{safe_model_name}_{args.domain}.jsonl"

    # 断点：启动时读取 checkpoint，恢复进度
    processed_count = 0
    last_item = None
    if checkpoint_file.exists():
        with open(checkpoint_file, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    last_item = json.loads(line)
                    processed_count += 1
    if last_item is not None:
        total_solved = last_item["Total solved"]
        total_executed = last_item["Total executed"]
        accuracy = total_solved / total_executed
        print(f"Checkpoint found! Restored {processed_count} records. "
              f"total_solved={total_solved}, accuracy={accuracy:.4f}")

    @retry(wait=wait_fixed(10), stop=stop_after_attempt(5))
    async def process_record(record, llm_name, mode):
        question = record["task"]
        roles = None
        constraint = ''
        manager = Manager(llm_name=llm_name)
        roles = await manager._act(question)
        for key, value in roles.items():
            roles[key]+=constraint
        response = await run_team_chat(question, llm_name, roles,args.domain)
        return response

    # 动态计算，适配任何数据集大小
    total = len(dataset)
    batch = args.batch_size
    num_batches = (total + batch - 1) // batch  # 向上取整

    # 断点：从上次中断的 batch 开始
    start_batch = processed_count // batch

    accuracy = total_solved / total_executed if total_executed > 0 else 0.0

    for i_batch in range(start_batch, num_batches):
        print(f"Batch {i_batch}",80*'-')
        start_ts = time.time()
        answer_log_probs = []
        answers = []

        current_batch = dataloader(dataset,args.batch_size,i_batch)
        if current_batch is None:
            print("No more data available.")
            break

        # 断点：跳过本 batch 内已处理的记录
        batch_start_idx = i_batch * batch
        records_to_process = []
        answers_to_process = []
        for j, record in enumerate(current_batch):
            global_idx = batch_start_idx + j
            if global_idx < processed_count:
                continue
            records_to_process.append(record)
            answers_to_process.append(record["answer"])

        if not records_to_process:
            continue

        tasks = [asyncio.create_task(process_record(r, args.llm_name, args.mode))
                 for r in records_to_process]
        raw_results = await asyncio.gather(*tasks)

        for task, answer, true_answer, in zip(records_to_process, raw_results, answers_to_process):
            predict_answer = gsm_get_predict(answer)
            is_solved = float(predict_answer)==float(true_answer)
            total_solved = total_solved + is_solved
            total_executed = total_executed + 1
            accuracy = total_solved/ total_executed
            updated_item = {
                "Question": task,
                "Answer": true_answer,
                "Response": answer,
                "Attempt answer": predict_answer,
                "Solved": is_solved,
                "Total solved": total_solved,
                "Total executed": total_executed,
                "Accuracy": accuracy
            }

            # 断点：将每条结果追加写入 checkpoint 文件
            with open(checkpoint_file, 'a', encoding='utf-8') as f:
                f.write(json.dumps(updated_item, ensure_ascii=False) + '\n')

        print(f"Batch time {time.time() - start_ts:.3f}")
        print(f"Accuracy: {accuracy}")

        print(f"Cost {Cost.instance().value}")
        print(f"PromptTokens {PromptTokens.instance().value}")
        print(f"CompletionTokens {CompletionTokens.instance().value}")


if __name__ == '__main__':
    asyncio.run(main())
