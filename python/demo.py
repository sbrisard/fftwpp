import pyfftwpp

if __name__ == "__main__":
    plan = pyfftwpp.Plan(1024)
    print(plan.cost())
    print(plan.flops())
