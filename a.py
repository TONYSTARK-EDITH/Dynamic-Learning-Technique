import asyncio

from sklearn.datasets import load_iris

from DLT import *


async def main():
    x, y = load_iris(return_X_y=True)
    for i in Utils.VALID_MODEL.value:
        refined = DLT(x, y, i(), is_trained=False, verbose=True)
        await refined.start()
        print(refined.refined_model.predict([[1, 2, 3, 4]]))
        print(refined)


if __name__ == "__main__":
    asyncio.run(main())
