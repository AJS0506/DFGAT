#!/usr/bin/env python3
"""
BestModel 디렉토리의 실험 결과를 파싱하여 CSV 파일로 정리하는 스크립트
"""

import os
import json
import csv
import re
from pathlib import Path
from collections import defaultdict

def parse_eval_file(filepath):
    """eval.txt 파일에서 파싱 가능한 데이터 추출"""
    results = {}
    
    if not os.path.exists(filepath):
        return results
    
    with open(filepath, 'r') as f:
        lines = f.readlines()
    
    for i, line in enumerate(lines):
        if line.startswith("# PARSEABLE_DATA_START_"):
            topk = int(line.split("_")[-1].strip())
            data = {}
            # 다음 7줄을 읽어서 파싱
            for j in range(i+1, min(i+8, len(lines))):
                if "=" in lines[j]:
                    key, value = lines[j].strip().split("=")
                    data[key] = float(value) if key != "topk" else int(value)
            if data:  # 데이터가 있으면 저장
                results[topk] = data
    
    # 파싱 가능한 데이터가 없으면 기존 형식 시도
    if not results:
        for i, line in enumerate(lines):
            if "[Top-K =" in line:
                topk_match = re.search(r'\[Top-K = (\d+)\]', line)
                if topk_match:
                    topk = int(topk_match.group(1))
                    data = {"topk": topk}
                    
                    # Recall, Precision, NDCG 라인 찾기
                    for j in range(i+1, min(i+10, len(lines))):
                        if "Recall:" in lines[j]:
                            # Macro와 Micro 값 파싱
                            parts = lines[j].replace("Recall:", "").strip().split()
                            if len(parts) >= 2:
                                data["macro_recall"] = float(parts[0])
                                data["micro_recall"] = float(parts[1])
                        elif "Precision:" in lines[j]:
                            parts = lines[j].replace("Precision:", "").strip().split()
                            if len(parts) >= 2:
                                data["macro_precision"] = float(parts[0])
                                data["micro_precision"] = float(parts[1])
                        elif "NDCG:" in lines[j]:
                            parts = lines[j].replace("NDCG:", "").strip().split()
                            if len(parts) >= 2:
                                data["macro_ndcg"] = float(parts[0])
                                data["micro_ndcg"] = float(parts[1])
                    
                    if len(data) > 1:  # topk 외에 다른 데이터가 있으면
                        results[topk] = data
    
    return results

def parse_config_file(filepath):
    """config.json 파일에서 실험 설정 정보 추출"""
    if not os.path.exists(filepath):
        return {}
    
    with open(filepath, 'r') as f:
        return json.load(f)

def collect_all_results(base_dir):
    """BestModel 디렉토리의 모든 실험 결과 수집"""
    all_results = []
    base_path = Path(base_dir)
    
    # 모든 실험 디렉토리 찾기
    for seed_dir in sorted(base_path.glob("*")):
        if not seed_dir.is_dir():
            continue
            
        seed = seed_dir.name
        
        for dataset_dir in sorted(seed_dir.glob("*")):
            if not dataset_dir.is_dir():
                continue
                
            dataset = dataset_dir.name
            
            for exp_dir in sorted(dataset_dir.glob("*")):
                if not exp_dir.is_dir():
                    continue
                
                # config.json 파싱
                config_path = exp_dir / "config.json"
                config = parse_config_file(config_path)
                
                # eval.txt 파싱
                eval_path = exp_dir / "eval.txt"
                eval_results = parse_eval_file(eval_path)
                
                # 결과 정리
                for topk, metrics in eval_results.items():
                    result = {
                        "seed": seed,
                        "dataset": dataset,
                        "exp_name": exp_dir.name,
                        "model": config.get("model", "Unknown"),
                        "model_type": config.get("model_type", "Unknown"),
                        "learning_rate": config.get("learning_rate", ""),
                        "embedding_dim": config.get("embedding_dim", ""),
                        "first_layer_dim": config.get("first_layer_dim", ""),
                        "second_layer_dim": config.get("second_layer_dim", ""),
                        "topk": topk,
                        "macro_recall": metrics.get("macro_recall", ""),
                        "macro_precision": metrics.get("macro_precision", ""),
                        "macro_ndcg": metrics.get("macro_ndcg", ""),
                        "micro_recall": metrics.get("micro_recall", ""),
                        "micro_precision": metrics.get("micro_precision", ""),
                        "micro_ndcg": metrics.get("micro_ndcg", ""),
                    }
                    all_results.append(result)
    
    return all_results


def write_combined_csv(results, output_file):
    """모든 결과를 하나의 CSV 파일로 저장"""
    # 시드 -> 데이터셋 -> 모델 -> Top-K 순으로 정렬
    results.sort(key=lambda x: (x["seed"], x["dataset"], x["model"], x["topk"]))
    
    fieldnames = [
        "seed", "dataset", "model", "model_type", "topk",
        "macro_recall", "macro_precision", "macro_ndcg",
        "micro_recall", "micro_precision", "micro_ndcg",
        "learning_rate", "embedding_dim", "first_layer_dim", "second_layer_dim",
        "exp_name"
    ]
    
    with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        
        for result in results:
            writer.writerow(result)
    
    print(f"Created combined CSV: {output_file}")

def write_pivot_csv_single(results, output_dir, metric_type="macro"):
    """@20, @40에 대한 Recall, Precision, NDCG 비교 테이블 생성 (macro 또는 micro)"""
    # 시드별, 데이터셋별로 그룹화
    pivot_data = defaultdict(lambda: defaultdict(dict))
    
    # @20, @40만 필터링하여 데이터 수집
    target_topks = [20, 40]
    
    for result in results:
        if result["topk"] not in target_topks:
            continue
            
        key = (result["seed"], result["dataset"], result["model"])
        topk = result["topk"]
        
        # Recall, Precision, NDCG @20, @40 값 저장
        pivot_data[key][f"R@{topk}"] = result.get(f"{metric_type}_recall", "")
        pivot_data[key][f"P@{topk}"] = result.get(f"{metric_type}_precision", "")
        pivot_data[key][f"N@{topk}"] = result.get(f"{metric_type}_ndcg", "")
    
    # 평균 계산을 위한 데이터 수집
    avg_data = defaultdict(lambda: defaultdict(list))
    
    for (seed, dataset, model), metrics in pivot_data.items():
        for metric_col in metrics:
            if metrics[metric_col] != "":
                avg_data[(dataset, model)][metric_col].append(metrics[metric_col])
    
    # 피벗 테이블 CSV 생성
    output_file = f"{output_dir}/results_pivot_{metric_type}.csv"
    
    # @20, @40에 대한 R, P, N 컬럼
    metric_columns = []
    for k in target_topks:
        metric_columns.extend([f"R@{k}", f"P@{k}", f"N@{k}"])
    
    fieldnames = ["seed", "dataset", "model"] + metric_columns
    
    with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        
        # 개별 시드 결과 작성
        for (seed, dataset, model), metrics in sorted(pivot_data.items()):
            row = {
                "seed": seed,
                "dataset": dataset,
                "model": model
            }
            row.update(metrics)
            writer.writerow(row)
        
        # 빈 줄 추가
        writer.writerow({})
        
        # 평균값 작성
        writer.writerow({"seed": "=== AVERAGE ===", "dataset": "", "model": ""})
        for (dataset, model), avg_metrics in sorted(avg_data.items()):
            row = {
                "seed": "AVG",
                "dataset": dataset,
                "model": model
            }
            for metric_col in metric_columns:
                if metric_col in avg_metrics and avg_metrics[metric_col]:
                    row[metric_col] = round(sum(avg_metrics[metric_col]) / len(avg_metrics[metric_col]), 6)
                else:
                    row[metric_col] = ""
            writer.writerow(row)
        
        # 빈 줄 추가
        writer.writerow({})
        
        # 표준편차 작성
        writer.writerow({"seed": "=== STD DEV ===", "dataset": "", "model": ""})
        for (dataset, model), avg_metrics in sorted(avg_data.items()):
            row = {
                "seed": "STD",
                "dataset": dataset,
                "model": model
            }
            for metric_col in metric_columns:
                if metric_col in avg_metrics and len(avg_metrics[metric_col]) > 1:
                    values = avg_metrics[metric_col]
                    mean = sum(values) / len(values)
                    variance = sum((x - mean) ** 2 for x in values) / len(values)
                    std_dev = variance ** 0.5
                    row[metric_col] = round(std_dev, 6)
                else:
                    row[metric_col] = ""
            writer.writerow(row)
    
    print(f"Created {metric_type.upper()} pivot table (@20, @40 R/P/N): {output_file}")

def write_pivot_csv(results, output_dir):
    """macro와 micro 피벗 테이블을 별도 파일로 생성"""
    write_pivot_csv_single(results, output_dir, "macro")
    write_pivot_csv_single(results, output_dir, "micro")

def main():
    # 경로 설정
    base_dir = "/home/jinsoo/DFGAT_NEW/B_DFGAT/BestModel"
    output_dir = "/home/jinsoo/DFGAT_NEW/B_DFGAT/results"
    
    # 출력 디렉토리 생성
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Collecting results from: {base_dir}")
    
    # 모든 결과 수집
    all_results = collect_all_results(base_dir)
    
    if not all_results:
        print("No results found!")
        return
    
    print(f"Found {len(all_results)} result entries")
    
    # 통합 CSV 생성
    write_combined_csv(all_results, f"{output_dir}/all_results.csv")
    
    # 피벗 테이블 CSV 생성
    write_pivot_csv(all_results, output_dir)
    
    print(f"\nAll CSV files have been created in: {output_dir}")
    print("Files created:")
    print("  - all_results.csv (combined)")
    print("  - results_pivot_macro.csv (MACRO R/P/N @20, @40)")
    print("  - results_pivot_micro.csv (MICRO R/P/N @20, @40)")

if __name__ == "__main__":
    main()