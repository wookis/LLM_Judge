from pathlib import Path
import pandas as pd
import uuid
import json

def get_target_data(input_dir: str) -> pd.DataFrame:
    list_root_dir = input_dir
    taget_file = {}

    for file in list_root_dir.glob('702*.csv'):
        print(file)
        model = file.stem.split('eval_')[1].split('.csv')[0]
        taget_file[model] = pd.read_csv(file)
        print(model, " : ", len(taget_file[model]))
        print("--------------------------------")
        
    # list(sorted(dict_llm2df.keys()))


    # dict_llm2df['midm-mini-inst-2.3.1'].head(3)

def get_target_file_to_dict(input_file: str) -> dict:

    target_dict = {}
    df = pd.read_csv(input_file)
    df = pd.DataFrame(df)
    print(df.head(3))

    for index, row in df.iterrows():
        id = uuid.uuid4()

        target_dict[id] = {
            'task': row['task'],
            'level': row['level'],
            'domain': row['domain'],
            'messages': row['messages'],
        }

    return target_dict






# def parse_csv_file(file_path: str) -> List[Dict[str, Any]]:
#     """
#     CSV 파일을 파싱하여 messages의 content를 추출합니다.
    
#     Args:
#         file_path (str): 파싱할 CSV 파일 경로
        
#     Returns:
#         List[Dict[str, Any]]: 파싱된 데이터 리스트
#     """
#     try:
#         # CSV 파일 읽기
#         df = pd.read_csv(file_path, encoding='utf-8')
#         logger.info(f"CSV 파일 로드 완료: {file_path}, 컬럼: {list(df.columns)}")
        
#         parsed_data = []
        
#         # 컬럼명 확인 및 content 컬럼 찾기
#         content_columns = []
#         for col in df.columns:
#             if 'content' in col.lower() or 'message' in col.lower() or 'text' in col.lower():
#                 content_columns.append(col)
        
#         if not content_columns:
#             # content 관련 컬럼이 없으면 첫 번째 텍스트 컬럼 사용
#             for col in df.columns:
#                 if df[col].dtype == 'object':  # 문자열 타입 컬럼
#                     content_columns.append(col)
#                     break
        
#         if not content_columns:
#             logger.warning(f"CSV 파일에서 content 컬럼을 찾을 수 없습니다: {file_path}")
#             return []
        
#         logger.info(f"사용할 content 컬럼: {content_columns}")
        
#         # 각 행 처리
#         for index, row in df.iterrows():
#             for content_col in content_columns:
#                 if pd.notna(row[content_col]) and str(row[content_col]).strip():
#                     content_value = str(row[content_col]).strip()
                    
#                     # 메타데이터 추출
#                     metadata = {}
#                     for col in df.columns:
#                         if col != content_col and pd.notna(row[col]):
#                             metadata[col] = row[col]
                    
#                     parsed_data.append({
#                         'file_path': file_path,
#                         'row_index': index,
#                         'message_content': content_value,
#                         'content_column': content_col,
#                         'metadata': metadata
#                     })
        
#         logger.info(f"CSV 파일 {file_path}에서 {len(parsed_data)}개의 메시지 추출")
#         return parsed_data
        
#     except Exception as e:
#         logger.error(f"CSV 파일 {file_path} 파싱 중 오류 발생: {e}")
#         return []


# def parse_tldc_file(file_path: str) -> List[Dict[str, Any]]:
#     """
#     TLDC 데이터 파일을 파싱하여 messages의 content를 추출합니다.
#     파일 확장자에 따라 적절한 파서를 선택합니다.
    
#     Args:
#         file_path (str): 파싱할 파일 경로
        
#     Returns:
#         List[Dict[str, Any]]: 파싱된 데이터 리스트
#     """
#     file_ext = Path(file_path).suffix.lower()
    
#     if file_ext == '.csv':
#         return parse_csv_file(file_path)
#     elif file_ext == '.json':
#         return parse_json_file(file_path)
#     else:
#         logger.warning(f"지원하지 않는 파일 형식입니다: {file_ext}")
#         return []


# def parse_json_file(file_path: str) -> List[Dict[str, Any]]:
#     """
#     JSON 파일을 파싱하여 messages의 content를 추출합니다.
    
#     Args:
#         file_path (str): 파싱할 JSON 파일 경로
        
#     Returns:
#         List[Dict[str, Any]]: 파싱된 데이터 리스트
#     """
#     try:
#         with open(file_path, 'r', encoding='utf-8') as f:
#             data = json.load(f)
        
#         parsed_data = []
        
#         # 데이터 구조에 따라 messages content 추출
#         if isinstance(data, list):
#             # 리스트 형태의 데이터
#             for item in data:
#                 if isinstance(item, dict) and 'messages' in item:
#                     messages = item['messages']
#                     if isinstance(messages, list):
#                         for msg in messages:
#                             if isinstance(msg, dict) and 'content' in msg:
#                                 parsed_data.append({
#                                     'file_path': file_path,
#                                     'message_content': msg['content'],
#                                     'message_role': msg.get('role', 'unknown'),
#                                     'timestamp': msg.get('timestamp', ''),
#                                     'metadata': {k: v for k, v in msg.items() 
#                                                if k not in ['content', 'role', 'timestamp']}
#                                 })
#         elif isinstance(data, dict):
#             # 딕셔너리 형태의 데이터
#             if 'messages' in item:
#                 messages = data['messages']
#                 if isinstance(messages, list):
#                     for msg in messages:
#                         if isinstance(msg, dict) and 'content' in msg:
#                             parsed_data.append({
#                                 'file_path': file_path,
#                                 'message_content': msg['content'],
#                                 'message_role': msg.get('role', 'unknown'),
#                                 'timestamp': msg.get('timestamp', ''),
#                                 'metadata': {k: v for k, v in msg.items() 
#                                            if k not in ['content', 'role', 'timestamp']}
#                             })
        
#         logger.info(f"JSON 파일 {file_path}에서 {len(parsed_data)}개의 메시지 추출")
#         return parsed_data
        
#     except Exception as e:
#         logger.error(f"JSON 파일 {file_path} 파싱 중 오류 발생: {e}")
#         return []


# def save_response_to_file(response_data: Dict[str, Any], output_dir: str, 
#                          original_filename: str) -> str:
#     """
#     응답 데이터를 파일로 저장합니다.
    
#     Args:
#         response_data (Dict[str, Any]): 저장할 응답 데이터
#         output_dir (str): 출력 디렉토리 경로
#         original_filename (str): 원본 파일명
        
#     Returns:
#         str: 저장된 파일 경로
#     """
#     try:
#         # 출력 디렉토리 생성
#         os.makedirs(output_dir, exist_ok=True)
        
#         # 파일명 생성 (원본 파일명에 _response 추가)
#         base_name = Path(original_filename).stem
#         output_filename = f"{base_name}_response.json"
#         output_path = os.path.join(output_dir, output_filename)
        
#         # 응답 데이터에 메타데이터 추가
#         response_data['metadata'] = {
#             'original_file': original_filename,
#             'processed_at': str(Path.cwd()),
#             'version': '1.0'
#         }
        
#         # JSON 파일로 저장
#         with open(output_path, 'w', encoding='utf-8') as f:
#             json.dump(response_data, f, ensure_ascii=False, indent=2)
        
#         logger.info(f"응답이 저장되었습니다: {output_path}")
#         return output_path
        
#     except Exception as e:
#         logger.error(f"응답 저장 중 오류 발생: {e}")
#         return ""

