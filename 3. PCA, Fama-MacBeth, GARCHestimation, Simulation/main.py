# -*- coding: utf-8 -*-
"""
PCA, Fama-MacBeth, GARCH, Simulation 파이프라인
"""

import sys


def print_menu():
    """메뉴 출력"""
    print("\n" + "=" * 60)
    print("PCA + Fama-MacBeth + GARCH 시뮬레이션")
    print("=" * 60)
    print("\n[개별 실행]")
    print("  1. PCA + Fama-MacBeth 분석")
    print("  2. t-GJR-GARCH + Monte Carlo 시뮬레이션")
    print("\n[일괄 실행]")
    print("  3. 전체 실행 (1 → 2)")
    print("\n  0. 종료")
    print("=" * 60)


def run_step1():
    """1단계: PCA + Fama-MacBeth"""
    print("\n" + "=" * 70)
    print("  [1단계] PCA + Fama-MacBeth 분석")
    print("=" * 70)

    from importlib import import_module
    module = import_module("1_PCA_Fama-MacBeth")
    module.run_pca_fm_analysis()


def run_step2():
    """2단계: GARCH + Monte Carlo"""
    print("\n" + "=" * 70)
    print("  [2단계] t-GJR-GARCH + Monte Carlo 시뮬레이션")
    print("=" * 70)

    from importlib import import_module
    module = import_module("2_t-GJR-GARCH_MonteCarloSimulation")
    module.run_simulation()


def run_all():
    """전체 실행 (1 → 2)"""
    print("\n[전체 실행]")
    run_step1()
    run_step2()


def execute_choice(choice):
    """선택된 작업 실행"""
    if choice == '1':
        run_step1()
    elif choice == '2':
        run_step2()
    elif choice == '3':
        run_all()
    else:
        print("\n잘못된 입력입니다.")
        return False

    print("\n" + "#" * 70)
    print("#  작업 완료")
    print("#" * 70)
    return True


def main():
    """메인 함수"""
    if len(sys.argv) > 1:
        choice = sys.argv[1]
        execute_choice(choice)
    else:
        while True:
            print_menu()
            choice = input("\n실행할 작업 번호를 입력하세요: ").strip()

            if choice == '0':
                print("\n프로그램을 종료합니다.")
                break

            execute_choice(choice)
            input("\n계속하려면 Enter를 누르세요...")


if __name__ == "__main__":
    main()
