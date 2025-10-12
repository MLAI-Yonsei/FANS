import os
import numpy as np
from pathlib import Path
import glob
import shutil

# 전역 시드 설정
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

def subsample_data(input_dir, output_dir, n_samples=1000):
    """
    데이터 파일들을 서브샘플링하여 새로운 디렉토리에 저장합니다.
    
    Parameters:
    input_dir (str): 원본 데이터가 있는 디렉토리 경로
    output_dir (str): 서브샘플링된 데이터를 저장할 디렉토리 경로
    n_samples (int): 추출할 샘플 수 (기본값: 1000)
    """
    
    # 출력 디렉토리 생성
    os.makedirs(output_dir, exist_ok=True)
    
    # 모든 노드 수에 대해 반복
    node_dirs = [d for d in os.listdir(input_dir) if d.startswith('nodes_')]
    
    for node_dir in node_dirs:
        node_path = os.path.join(input_dir, node_dir)
        if not os.path.isdir(node_path):
            continue
            
        print(f"처리 중: {node_dir}")
        
        # 각 노드 디렉토리 내의 그래프 타입들 (ER, SF 등)
        graph_types = [d for d in os.listdir(node_path) if os.path.isdir(os.path.join(node_path, d))]
        
        for graph_type in graph_types:
            graph_path = os.path.join(node_path, graph_type)
            print(f"  처리 중: {graph_type}")
            
            # 출력 디렉토리 구조 생성
            output_graph_path = os.path.join(output_dir, node_dir, graph_type)
            os.makedirs(output_graph_path, exist_ok=True)
            
            # 모든 파일들 가져오기
            all_files = [f for f in os.listdir(graph_path) if os.path.isfile(os.path.join(graph_path, f))]
            
            # 파일 분류 및 처리
            for filename in all_files:
                file_path = os.path.join(graph_path, filename)
                
                # data_env1_*.npy와 data_env2_*.npy 파일들은 서브샘플링
                if filename.startswith('data_env1_') and filename.endswith('.npy'):
                    process_file(file_path, output_graph_path, n_samples)
                elif filename.startswith('data_env2_') and filename.endswith('.npy'):
                    process_file(file_path, output_graph_path, n_samples)
                else:
                    # 나머지 파일들은 그대로 복사
                    copy_file(file_path, output_graph_path)

def process_file(file_path, output_dir, n_samples):
    """
    개별 파일을 처리하여 서브샘플링합니다.
    
    Parameters:
    file_path (str): 처리할 파일 경로
    output_dir (str): 출력 디렉토리
    n_samples (int): 추출할 샘플 수
    """
    try:
        # 파일명 추출
        filename = os.path.basename(file_path)
        
        # 데이터 로드
        data = np.load(file_path)
        print(f"    서브샘플링: {filename} (원본 크기: {data.shape})")
        
        # 샘플 수가 요청된 수보다 적으면 전체 데이터 사용
        if len(data) <= n_samples:
            subsampled_data = data
            print(f"      경고: {filename}의 샘플 수({len(data)})가 요청된 수({n_samples})보다 적습니다. 전체 데이터를 사용합니다.")
        else:
            # 무작위로 n_samples개 선택
            indices = np.random.choice(len(data), n_samples, replace=False)
            subsampled_data = data[indices]
        
        # 결과 저장
        output_path = os.path.join(output_dir, filename)
        np.save(output_path, subsampled_data)
        print(f"      저장 완료: {output_path} (새로운 크기: {subsampled_data.shape})")
        
    except Exception as e:
        print(f"      오류 발생 {filename}: {str(e)}")

def copy_file(file_path, output_dir):
    """
    파일을 그대로 복사합니다.
    
    Parameters:
    file_path (str): 복사할 파일 경로
    output_dir (str): 출력 디렉토리
    """
    try:
        # 파일명 추출
        filename = os.path.basename(file_path)
        
        # 출력 경로 설정
        output_path = os.path.join(output_dir, filename)
        
        # 파일 복사
        shutil.copy2(file_path, output_path)
        print(f"    복사 완료: {filename}")
        
    except Exception as e:
        print(f"      복사 오류 {filename}: {str(e)}")

def main():
    """
    메인 실행 함수
    """
    # 시드 재설정 (명시적으로 확인)
    np.random.seed(RANDOM_SEED)
    
    # 경로 설정
    input_dir = 'data'
    output_dir = 'data_new'
    n_samples = 1000
    
    print(f"데이터 처리 시작...")
    print(f"입력 디렉토리: {input_dir}")
    print(f"출력 디렉토리: {output_dir}")
    print(f"서브샘플링할 샘플 수: {n_samples}")
    print(f"랜덤 시드: {RANDOM_SEED}")
    print("처리 방식:")
    print("  - data_env1_*.npy, data_env2_*.npy: 서브샘플링")
    print("  - 나머지 파일들: 그대로 복사")
    print("-" * 50)
    
    # 서브샘플링 실행
    subsample_data(input_dir, output_dir, n_samples)
    
    print("-" * 50)
    print("데이터 처리 완료!")

if __name__ == "__main__":
    main()