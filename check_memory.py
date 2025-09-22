#!/usr/bin/env python3
"""
메모리 상태 확인 및 최적화 스크립트
"""
import psutil
import gc
import os

def check_system_memory():
    """시스템 메모리 상태 확인"""
    memory = psutil.virtual_memory()
    print(f"💾 시스템 메모리 상태:")
    print(f"   총 메모리: {memory.total / 1024 / 1024 / 1024:.1f} GB")
    print(f"   사용 가능: {memory.available / 1024 / 1024 / 1024:.1f} GB")
    print(f"   사용률: {memory.percent:.1f}%")
    print(f"   사용 중: {memory.used / 1024 / 1024 / 1024:.1f} GB")
    
    if memory.percent > 80:
        print("⚠️ 경고: 메모리 사용률이 80%를 초과했습니다!")
        return False
    elif memory.percent > 60:
        print("⚠️ 주의: 메모리 사용률이 60%를 초과했습니다.")
        return True
    else:
        print("✅ 메모리 상태 양호")
        return True

def check_process_memory():
    """현재 프로세스 메모리 사용량 확인"""
    try:
        process = psutil.Process()
        memory_info = process.memory_info()
        rss_mb = memory_info.rss / 1024 / 1024
        vms_mb = memory_info.vms / 1024 / 1024
        
        print(f"🔍 현재 프로세스 메모리:")
        print(f"   RSS: {rss_mb:.1f} MB")
        print(f"   VMS: {vms_mb:.1f} MB")
        
        if rss_mb > 500:
            print("⚠️ 경고: 프로세스 메모리 사용량이 높습니다!")
            return False
        else:
            print("✅ 프로세스 메모리 상태 양호")
            return True
    except Exception as e:
        print(f"❌ 프로세스 메모리 확인 실패: {e}")
        return False

def optimize_memory():
    """메모리 최적화 실행"""
    print("🧹 메모리 최적화 실행...")
    
    # 가비지 컬렉션
    collected = gc.collect()
    print(f"   가비지 컬렉션: {collected}개 객체 정리")
    
    # 메모리 상태 재확인
    system_ok = check_system_memory()
    process_ok = check_process_memory()
    
    return system_ok and process_ok

def main():
    """메인 함수"""
    print("🔍 메모리 상태 진단 시작\n")
    
    # 1. 시스템 메모리 확인
    system_ok = check_system_memory()
    print()
    
    # 2. 프로세스 메모리 확인
    process_ok = check_process_memory()
    print()
    
    # 3. 메모리 최적화 필요 여부 판단
    if not system_ok or not process_ok:
        print("🔧 메모리 최적화가 필요합니다.")
        optimize_memory()
    else:
        print("✅ 메모리 상태가 양호합니다.")
    
    # print("\n📋 권장사항:")
    # print("1. Cursor AI를 재시작하세요")
    # print("2. main_safe.py를 사용하여 안전 모드로 실행하세요")
    # print("3. 청크 크기를 더 작게 조정하세요 (chunk_size=10)")
    # print("4. 다른 메모리 사용량이 높은 프로그램을 종료하세요")

if __name__ == "__main__":
    main()
