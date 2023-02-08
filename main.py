import pandas as pd


def main():
    df = pd.read_csv('DATA.csv', sep=';')

    # STUDENT ID and COURSE ID aren't atrributes
    attributes = df[df.columns[~df.columns.isin(['STUDENT ID', 'COURSE ID', 'GRADE'])]]

    # get sqrt(n) (sqrt(30) +-= 5) random attributes
    # im too lazy so here is numbers : 3 10 18 24 23
    attributes_to_choose = ['3', '10', '18', '23', '24']

    # choosing 5 attributes
    attributes = df[df.columns[df.columns.isin(attributes_to_choose)]]

    # normilize grades with rule: 1,2 - 0 (bad) and 3,4,5 - 1 (good)
    grades = df['GRADE'].map(lambda x: 0 if x <= 2 else 1)
    print(attributes)
    print(grades)


if __name__ == '__main__':
    main()
