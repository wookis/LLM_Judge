import os
import json
import glob
import asyncio
import pandas as pd
from pathlib import Path
from typing import Dict, List, Any, Optional
from llm_judge import llm_interfaces
from llm_judge.llm_interfaces import OpenAILLM, KT_MAGMA_DEV_LLM
from llm_judge.core import LLM_MODEL
from dotenv import load_dotenv
from utils.logger import logger

from utils.parser import get_target_file_to_dict  
from mmo_lr_utils.labeler.requester import get_dict_response_from_body  

load_dotenv()

import ssl
def _allowSelfSignedHttps(allowed):
    if (
        allowed
        and not os.environ.get("PYTHONHTTPSVERIFY", "")
        and getattr(ssl, "_create_unverified_context", None)
    ):
        ssl._create_default_https_context = ssl._create_unverified_context


_allowSelfSignedHttps(True)




def get_response(content: str) -> str:
    """
    주어진 content에 대한 응답을 생성합니다.
    실제 구현에서는 LLM API 호출이나 다른 로직을 사용할 수 있습니다.
    
    Args:
        content (str): 입력 content
        
    Returns:
        str: 생성된 응답
    """
    # TODO: 실제 LLM API 호출이나 응답 생성 로직 구현
    # 현재는 예시 응답을 반환합니다
    return f"응답: {content[:100]}..."

def process_tldc_files(input_dir: str, output_dir: str) -> Dict[str, Any]:
    """
    TLDC 데이터 디렉토리의 모든 파일을 처리합니다.
    
    Args:
        input_dir (str): 입력 디렉토리 경로
        output_dir (str): 출력 디렉토리 경로
        
    Returns:
        Dict[str, Any]: 처리 결과 요약
    """
    # 입력 디렉토리 확인
    if not os.path.exists(input_dir):
        logger.error(f"입력 디렉토리가 존재하지 않습니다: {input_dir}")
        return {}
    
    # 모든 CSV와 JSON 파일 찾기
    csv_files = glob.glob(os.path.join(input_dir, "*.csv"))
    json_files = glob.glob(os.path.join(input_dir, "*.json"))
    all_files = csv_files + json_files
    
    if not all_files:
        logger.warning(f"입력 디렉토리에 CSV 또는 JSON 파일이 없습니다: {input_dir}")
        return {}
    
    logger.info(f"총 {len(all_files)}개의 파일을 처리합니다. (CSV: {len(csv_files)}, JSON: {len(json_files)})")
    
    # 처리 결과 요약
    summary = {
        'total_files': len(all_files),
        'csv_files': len(csv_files),
        'json_files': len(json_files),
        'processed_files': 0,
        'total_messages': 0,
        'total_responses': 0,
        'errors': [],
        'output_files': []
    }
    
    # 각 파일 처리
    for file_path in all_files:
        try:
            logger.info(f"파일 처리 중: {file_path}")
            
            # 파일 파싱
            parsed_data = parse_tldc_file(file_path)
            if not parsed_data:
                logger.warning(f"파일에서 메시지를 추출할 수 없습니다: {file_path}")
                continue
            
            summary['total_messages'] += len(parsed_data)
            
            # 각 메시지에 대해 응답 생성
            file_responses = []
            for msg_data in parsed_data:
                try:
                    # get_response 호출
                    response_content = get_response(msg_data['message_content'])
                    
                    # 응답 데이터 구성
                    response_data = {
                        'original_message': msg_data,
                        'response_content': response_content,
                        'response_timestamp': str(Path.cwd())
                    }
                    
                    file_responses.append(response_data)
                    summary['total_responses'] += 1
                    
                except Exception as e:
                    error_msg = f"메시지 응답 생성 중 오류: {e}"
                    logger.error(error_msg)
                    summary['errors'].append(error_msg)
            
            # 파일별로 응답 저장
            if file_responses:
                output_path = save_response_to_file(
                    {'responses': file_responses}, 
                    output_dir, 
                    os.path.basename(file_path)
                )
                if output_path:
                    summary['output_files'].append(output_path)
                    summary['processed_files'] += 1
            
        except Exception as e:
            error_msg = f"파일 {file_path} 처리 중 오류: {e}"
            logger.error(error_msg)
            summary['errors'].append(error_msg)
    
    return summary

tagetLLM = LLM_MODEL()

def main():
    """메인 실행 함수"""
    # 디렉토리 경로 설정
    current_dir = Path(__file__).parent.parent
    input_dir = current_dir / "dataset" / "1.tldc_data"
    input_file = current_dir / "dataset" / "1.tldc_data" / "702-samples-midm-mini.csv"
    output_dir = current_dir / "dataset" / "2.answer"
    
    logger.info("TLDC 데이터 처리 시작")
    logger.info(f"입력 디렉토리: {input_dir}")
    logger.info(f"출력 디렉토리: {output_dir}")

    try:   
    # 모델 연동
        gpt4_o = OpenAILLM(model_name="gpt-4o")
        gemma_2_9b_it = KT_MAGMA_DEV_LLM(model_name="gemma-2-9b-it")

        #tagetLLM.add_model(gpt4_o)
        tagetLLM.add_model(gemma_2_9b_it)
    except Exception as e:
        print(f"오류: {e}")

    taget_data = get_target_file_to_dict(input_file)

    

    print(taget_data[list(taget_data.keys())[0]]['messages'])
    print(taget_data[list(taget_data.keys())[0]]['task'])
    print(taget_data[list(taget_data.keys())[0]]['level'])
    print(taget_data[list(taget_data.keys())[0]]['domain'])
    print(len(taget_data))

    messsages = taget_data[list(taget_data.keys())[0]]['messages']
    print(messsages)
    print(messsages[1]['content'])

    for model in tagetLLM.models:
        print(model)
        response = tagetLLM.models[model].generate_response(messsages)
        print(response)
        print("--------------------------------")



    #print(tagetLLM.models)
    #response = get_response_from_llm(taget_data[list(taget_data.keys())[0]]['model'], messsages)
    # response = tagetLLM.models[taget_data[list(taget_data.keys())[0]]['model']].generate_response(messsages)
    # print(response)

    # response = llm_interfaces.KT_MAGMA_DEV_LLM(model_name="gemma-2-9b-it").generate_response(messsages)
    # print(response)


    # user_content = []  
    # for message in messsages:
    #     if message['role'] == 'user':
    #         user_content.append(message['content'])
    # print("User content:", user_content)
    # #[{'role': 'system', 'content': ''}, {'role': 'user', 'content': '라인하르트 자프티히는 독일의 축구 감독이야.'}]
 


    # for model in tagetLLM.models:
    #     logger.info(f"TagetModel: {model}")
    #     logger.info(tagetLLM.models[model].generate_response("Hello, how are you?"))  
    # #get_response_from_llm("gpt-4o", prompt)




if __name__ == "__main__":
    main() 