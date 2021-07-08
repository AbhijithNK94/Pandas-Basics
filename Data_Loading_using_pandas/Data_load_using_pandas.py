import pandas as pd
import numpy as np

'''File Formats:
.txt
.csv
.xls
.xlsx
.json
and more....
'''
'''                                     
                   LOADING txt files.
'''
# text = pd.read_table("TXT.txt")
'''
 id,"name","age","salary"
0       1,"devin",24,10000
1      2,"anjana",25,20000
2         3,"don",21,80000 
'''
# print(text, '\n')
# print(type(text), '\n')  # The datatype will be of dataframe model of pandas.
# print(text.columns, '\n')
''' Index(['id,"name","age","salary"'], dtype='object') '''
# print(len(text.columns), '\n')  # The length of column will be 1 as because it is a string format.

# Hence we gonna use delimiter.
# text = pd.read_table("TXT.txt", sep=",")
# print(text, '\n')
'''
id    name  age  salary
0   1   devin   24   10000
1   2  anjana   25   20000
2   3     don   21   80000

'''
# print(text.columns)
'''Index(['id', 'name', 'age', 'salary'], dtype='object')'''
# print(len(text.columns))  # len(columns) will be 4.
# text.set_index('id', inplace=True)
# print(text)
# print(text.values)

'''CSV- Comma sep values
Since our text file is CSV format, we don't need to pass separator arg as that for 
pd.read_table() fn.
'''
# df = pd.read_csv('TXT.txt')
# print(df)

'''
                      LOADING CSV files.
'''
# data = pd.read_csv("Salary_Data.csv")
# print(data, '\n')
# print(type(data), '\n')  # save as a dataframe
# print(data.describe(), '\n')  # to know basic fn values to get a summary.
# print(data.head())  # First 5 rows by def
# print(data.tail())  # last 5 rows by default
# print(data.head(10))  # First 10 rows
# print(data.columns)

# Find salary greater than 60000
# print(data.Salary > 60000)
# print(data[data.Salary > 60000])

# Change salary to 70000 if it is > 60000.
# data.loc[data.Salary>60000, 'Salary'] = 70000
# print(data)

# df = pd.read_table("TXT2.txt", sep=';')
# print(df)

# Writing to CSV file:
# data = {'Name':['Tom', 'Jack', 'Steve', 'Ricky'], 'Age':[28, 34, 29, 42]}
# df = pd.DataFrame(data)
# print(df, '\n')
# df.to_csv('data.csv', index=False)
'''
# Index = False is for avoiding default index while saving as a CSV file.
# If the index is not avoided, then while reading the csv to DF it will be doing
# second indexing.
'''
# s = pd.read_csv("data.csv")
# print(s, '\n')

# s = pd.read_csv("data.csv", header=None)
# print(s, '\n')
'''When we pass the arg header=None, then the Column names will not be considered as a 
header and the column indexing will be default as shown below.
       0    1
0   Name  Age
1    Tom   28
2   Jack   34
3  Steve   29
4  Ricky   42
'''
# s = pd.read_csv("data.csv", header=1)  # to consider 1st row as header/column name.
# print(s)
'''
     Tom  28
0   Jack  34
1  Steve  29
2  Ricky  42
'''

'''
                      Diff loading method: Using OS module.
We can change directory by os module. 
'''
# import os
# os.chdir('C:\\Users\\abhij\\PycharmProjects\\Python basics\\Datas')
# print(os.getcwd())
# s = pd.read_table('TXT5.txt', sep=';')
# '''We don't need to specify the path of the file in this example since
# we already changed the path of the directory to the respective directory
# were the data is present.'''
# print(s)

# EXAMPLE PROBLEM (CAR DATA):
# df = pd.read_csv("mtcars.csv")
'''
COLUMN DESCRIPTION:
# [, 1]	mpg	Miles/(US) gallon
# [, 2]	cyl	Number of cylinders
# [, 3]	disp	Displacement (cu.in.)
# [, 4]	hp	Gross horsepower
# [, 5]	drat	Rear axle ratio
# [, 6]	wt	Weight (1000 lbs)
# [, 7]	qsec	1/4 mile time
# [, 8]	vs	Engine (0 = V-shaped, 1 = straight)
# [, 9]	am	Transmission (0 = automatic, 1 = manual)
# [,10]	gear	Number of forward gears
# [,11]	carb	Number of carburetors
'''
# print(df, '\n')
# print(df.columns, '\n')
# df.set_index('Unnamed: 0', inplace=True)
# print(df, '\n')

# Q.1) Print car details if the mileage is between 15 and 20.
# df1 =df.loc[(df['mpg']>15) & (df['mpg']<20)]
# print(df1)

# Q.2) Print manual type car details with 6 cyl engine.
# df1 = df.loc[(df.am == 1) & (df.cyl == 6)]
# print(df1, '\n')

# Q.3) Change auto car parameter from '0' to '2'
# df1 = df.am.replace(0, 2)  # It doesn't modifies the existing dataframe.
# print(df1)
# df.loc[df.am == 0, "am"] = 2  # This is the suitable method to modify the existing dataframe
# print(df, '\n')

# Q.4) Add 2 to every data in "vs" column
# df.vs += 2
# print(df.vs)


'''
                            FETCHING DATA FROM XML FILES:
install xlrd package to load.
The latest version of xlrd(2.0.1) doesn't suits for certain xml files and hence we will install
the version xlrd(1.2.0)
'''
# df = pd.read_excel("mtcars.xlsx")
# print(df, '\n')
'''
O/P
            Unnamed: 0   mpg  cyl   disp   hp  ...   qsec  vs  am  gear  carb
0             Mazda RX4  21.0    6  160.0  110  ...  16.46   0   1     4     4
1         Mazda RX4 Wag  21.0    6  160.0  110  ...  17.02   0   1     4     4
2            Datsun 710  22.8    4  108.0   93  ...  18.61   1   1     4     1
3        Hornet 4 Drive  21.4    6  258.0  110  ...  19.44   1   0     3     1
4     Hornet Sportabout  18.7    8  360.0  175  ...  17.02   0   0     3     2
5               Valiant  18.1    6  225.0  105  ...  20.22   1   0     3     1
6            Duster 360  14.3    8  360.0  245  ...  15.84   0   0     3     4
7             Merc 240D  24.4    4  146.7   62  ...  20.00   1   0     4     2
8              Merc 230  22.8    4  140.8   95  ...  22.90   1   0     4     2
9              Merc 280  19.2    6  167.6  123  ...  18.30   1   0     4     4
10            Merc 280C  17.8    6  167.6  123  ...  18.90   1   0     4     4
11           Merc 450SE  16.4    8  275.8  180  ...  17.40   0   0     3     3
12           Merc 450SL  17.3    8  275.8  180  ...  17.60   0   0     3     3
13          Merc 450SLC  15.2    8  275.8  180  ...  18.00   0   0     3     3
14   Cadillac Fleetwood  10.4    8  472.0  205  ...  17.98   0   0     3     4
15  Lincoln Continental  10.4    8  460.0  215  ...  17.82   0   0     3     4
16    Chrysler Imperial  14.7    8  440.0  230  ...  17.42   0   0     3     4
17             Fiat 128  32.4    4   78.7   66  ...  19.47   1   1     4     1
18          Honda Civic  30.4    4   75.7   52  ...  18.52   1   1     4     2
19       Toyota Corolla  33.9    4   71.1   65  ...  19.90   1   1     4     1
20        Toyota Corona  21.5    4  120.1   97  ...  20.01   1   0     3     1
21     Dodge Challenger  15.5    8  318.0  150  ...  16.87   0   0     3     2
22          AMC Javelin  15.2    8  304.0  150  ...  17.30   0   0     3     2
23           Camaro Z28  13.3    8  350.0  245  ...  15.41   0   0     3     4
24     Pontiac Firebird  19.2    8  400.0  175  ...  17.05   0   0     3     2
25            Fiat X1-9  27.3    4   79.0   66  ...  18.90   1   1     4     1
26        Porsche 914-2  26.0    4  120.3   91  ...  16.70   0   1     5     2
27         Lotus Europa  30.4    4   95.1  113  ...  16.90   1   1     5     2
28       Ford Pantera L  15.8    8  351.0  264  ...  14.50   0   1     5     4
29         Ferrari Dino  19.7    6  145.0  175  ...  15.50   0   1     5     6
30        Maserati Bora  15.0    8  301.0  335  ...  14.60   0   1     5     8
31           Volvo 142E  21.4    4  121.0  109  ...  18.60   1   1     4     2

[32 rows x 12 columns] 

'''

# Removing heading, skipping first 3 rows, skipping last 3 rows:
# df = pd.read_excel("mtcars.xlsx", skiprows=3, skipfooter=3)
# print(df, '\n')
'''
O/P
   Datsun 710  22.8  4    108   93  ...  18.61  1  1.1  4.1  1.2
0        Hornet 4 Drive  21.4  6  258.0  110  ...  19.44  1    0    3    1
1     Hornet Sportabout  18.7  8  360.0  175  ...  17.02  0    0    3    2
2               Valiant  18.1  6  225.0  105  ...  20.22  1    0    3    1
3            Duster 360  14.3  8  360.0  245  ...  15.84  0    0    3    4
4             Merc 240D  24.4  4  146.7   62  ...  20.00  1    0    4    2
5              Merc 230  22.8  4  140.8   95  ...  22.90  1    0    4    2
6              Merc 280  19.2  6  167.6  123  ...  18.30  1    0    4    4
7             Merc 280C  17.8  6  167.6  123  ...  18.90  1    0    4    4
8            Merc 450SE  16.4  8  275.8  180  ...  17.40  0    0    3    3
9            Merc 450SL  17.3  8  275.8  180  ...  17.60  0    0    3    3
10          Merc 450SLC  15.2  8  275.8  180  ...  18.00  0    0    3    3
11   Cadillac Fleetwood  10.4  8  472.0  205  ...  17.98  0    0    3    4
12  Lincoln Continental  10.4  8  460.0  215  ...  17.82  0    0    3    4
13    Chrysler Imperial  14.7  8  440.0  230  ...  17.42  0    0    3    4
14             Fiat 128  32.4  4   78.7   66  ...  19.47  1    1    4    1
15          Honda Civic  30.4  4   75.7   52  ...  18.52  1    1    4    2
16       Toyota Corolla  33.9  4   71.1   65  ...  19.90  1    1    4    1
17        Toyota Corona  21.5  4  120.1   97  ...  20.01  1    0    3    1
18     Dodge Challenger  15.5  8  318.0  150  ...  16.87  0    0    3    2
19          AMC Javelin  15.2  8  304.0  150  ...  17.30  0    0    3    2
20           Camaro Z28  13.3  8  350.0  245  ...  15.41  0    0    3    4
21     Pontiac Firebird  19.2  8  400.0  175  ...  17.05  0    0    3    2
22            Fiat X1-9  27.3  4   79.0   66  ...  18.90  1    1    4    1
23        Porsche 914-2  26.0  4  120.3   91  ...  16.70  0    1    5    2
24         Lotus Europa  30.4  4   95.1  113  ...  16.90  1    1    5    2
25       Ford Pantera L  15.8  8  351.0  264  ...  14.50  0    1    5    4

[26 rows x 12 columns] 

'''

# Making 0th column as index, nrows is for how many rows should we load.
# Only loading 10 rows from the 1st row.
# df = pd.read_excel("mtcars.xlsx", index_col=0, nrows=10)
# print(df, '\n')
'''
O/P
                 mpg  cyl   disp   hp  drat  ...   qsec  vs  am  gear  carb
Mazda RX4          21.0    6  160.0  110  3.90  ...  16.46   0   1     4     4
Mazda RX4 Wag      21.0    6  160.0  110  3.90  ...  17.02   0   1     4     4
Datsun 710         22.8    4  108.0   93  3.85  ...  18.61   1   1     4     1
Hornet 4 Drive     21.4    6  258.0  110  3.08  ...  19.44   1   0     3     1
Hornet Sportabout  18.7    8  360.0  175  3.15  ...  17.02   0   0     3     2
Valiant            18.1    6  225.0  105  2.76  ...  20.22   1   0     3     1
Duster 360         14.3    8  360.0  245  3.21  ...  15.84   0   0     3     4
Merc 240D          24.4    4  146.7   62  3.69  ...  20.00   1   0     4     2
Merc 230           22.8    4  140.8   95  3.92  ...  22.90   1   0     4     2
Merc 280           19.2    6  167.6  123  3.92  ...  18.30   1   0     4     4

'''
# To write excel, install "openpyxl" package.
# df.to_excel("test.xlsx", sheet_name="data1", index=False)  # To create a new excel file from existing dataframe(df).

'''
                                DATA LOADING FROM json file.
'''
# data = pd.read_json("test2.json")
# print(type(data))
# print(data, '\n')

# Converting dataframe to json file
# df = pd.DataFrame([ ('bird', 'Falconiformes', 389.0),
#                     ('bird', 'Psittaciformes', 24.0),
#                     ('mammal', 'Carnivora', 80.2),
#                     ('mammal', 'Primates', np.nan),
#                     ('mammal', 'Carnivora', 58)      ],
#         index=['falcon', 'parrot', 'lion', 'monkey', 'leopard'],
#         columns=('class', 'order', 'max_speed'))
# print(df, '\n')
# df.to_json("df_to_json.json")
# data = pd.read_json("df_to_json.json")
# print(data)

# EXAMPLE PROBLEM:

# df = pd.read_csv("lung.csv")
"""
                                    DESCRIPTION OF DATA'S:
inst:	Institution code
time:	Survival time in days
status:	censoring status 1=censored, 2=dead
age:	Age in years
sex:	Male=1 Female=2
ph.ecog:	ECOG performance score (0=good 5=dead)
ph.karno:	Karnofsky performance score (bad=0-good=100) rated by physician
pat.karno:	Karnofsky performance score as rated by patient
meal.cal:	Calories consumed at meals
wt.loss:	Weight loss in last six months
"""
# print(df, '\n')
# print(df.shape, '\n')
# print(df.columns, '\n')
# print(df.dtypes, '\n')  # To get data types of all columns.
# df.set_index("Unnamed: 0", inplace=True)
# print(df.head(), '\n')
# print(df.sex, '\n')

# Reset the data to values 1 and 2 for column "sex" to male and female as given in the description.
# df.loc[df.sex == 1] = "male"
# df.loc[df.sex == 2] = "female"
# print(df.sex)

# Removing selective items from a column.
# Q) Remove the rows from the df with status == 1.
# Fetching the status == 1 data's from the dataframe and converting to list.
# l = df.loc[df.status == 1].index.tolist()  # Fetching the rows with status == 1 and converting it to list.
# print(l, '\n')
# df = df.drop(l)  # Dropping the list of index
# print(df.status, '\n')
# print(df, '\n')

# IMPORTING Data's from a github repository.
# data = pd.read_csv("https://raw.githubusercontent.com/mwaskom/seaborn-data/master/diamonds.csv")
# print(data, '\n')
# print(data.shape, '\n')
# print(data.head(), '\n')  # Will print first 5 rows by default.
# print(data.describe(), '\n')  # Statistical data's of each column.
# print(data.nunique(), '\n')
# print(data.color[0])

# Create a column "quality_color" by join "cut" column and "color" column.
# data["quality_color"] = data.cut+"-"+data.color
# print(data.head(), '\n')
'''
O/P
   carat      cut color clarity  depth  ...  price     x     y     z  quality_color
0   0.23    Ideal     E     SI2   61.5  ...    326  3.95  3.98  2.43        Ideal-E
1   0.21  Premium     E     SI1   59.8  ...    326  3.89  3.84  2.31      Premium-E
2   0.23     Good     E     VS1   56.9  ...    327  4.05  4.07  2.31         Good-E
3   0.29  Premium     I     VS2   62.4  ...    334  4.20  4.23  2.63      Premium-I
4   0.31     Good     J     SI2   63.3  ...    335  4.34  4.35  2.75         Good-J
'''
# data.pop("cut")
# data.pop("color")
# print(data.head(), '\n')
'''
O/P
carat clarity  depth  table  price     x     y     z quality_color
0   0.23     SI2   61.5   55.0    326  3.95  3.98  2.43       Ideal-E
1   0.21     SI1   59.8   61.0    326  3.89  3.84  2.31     Premium-E
2   0.23     VS1   56.9   65.0    327  4.05  4.07  2.31        Good-E
3   0.29     VS2   62.4   58.0    334  4.20  4.23  2.63     Premium-I
4   0.31     SI2   63.3   58.0    335  4.34  4.35  2.75        Good-J 
'''
# df = data.reindex(columns=['carat', 'quality_color', 'clarity', 'depth', 'table', 'price', 'x', 'y', 'z'])
# print(df.head())
'''
O/P
 carat quality_color clarity  depth  table  price     x     y     z
0   0.23       Ideal-E     SI2   61.5   55.0    326  3.95  3.98  2.43
1   0.21     Premium-E     SI1   59.8   61.0    326  3.89  3.84  2.31
2   0.23        Good-E     VS1   56.9   65.0    327  4.05  4.07  2.31
3   0.29     Premium-I     VS2   62.4   58.0    334  4.20  4.23  2.63
4   0.31        Good-J     SI2   63.3   58.0    335  4.34  4.35  2.75
'''

# EXAMPLE for pivoting (pivot_table):

# data = pd.read_excel("SaleData.xlsx")
# print(data.head(), '\n')
# print(data.columns, '\n')
'''
O/P:
  OrderDate   Region  Manager  ... Units Unit_price  Sale_amt
0 2018-01-06     East   Martha  ...  95.0     1198.0  113810.0
1 2018-01-23  Central  Hermann  ...  50.0      500.0   25000.0
2 2018-02-09  Central  Hermann  ...  36.0     1198.0   43128.0
3 2018-02-26  Central  Timothy  ...  27.0      225.0    6075.0
4 2018-03-15     West  Timothy  ...  56.0     1198.0   67088.0
.........
.........
.........
'''

# Group by region and manager
# df = pd.pivot_table(data, index=["Region","Manager"])
# print(df, '\n')

'''
O/P:
                    Sale_amt   Unit_price      Units
Region  Manager                                      
Central Douglas  41338.666667   607.666667  52.000000
        Hermann  30425.708333   496.083333  53.916667
        Martha   49922.500000  1023.500000  45.750000
        Timothy  28191.000000   724.200000  42.600000
East    Douglas  16068.000000   261.166667  56.666667
        Martha   27280.300000   496.300000  52.100000
West    Douglas  33418.000000   849.000000  44.500000
        Timothy  22015.750000   512.000000  35.500000

'''
# print(df.columns)
'''O/P
Index(['Sale_amt', 'Unit_price', 'Units'], dtype='object')
'''
# print(df.index)
'''O/P
MultiIndex([('Central', 'Douglas'),
            ('Central', 'Hermann'),
            ('Central',  'Martha'),
            ('Central', 'Timothy'),
            (   'East', 'Douglas'),
            (   'East',  'Martha'),
            (   'West', 'Douglas'),
            (   'West', 'Timothy')],
           names=['Region', 'Manager'])
'''
# Group by Condition:
# print(data.query('Region == ["West"]'), '\n')
'''
O/P:
  OrderDate Region  Manager  ... Units Unit_price  Sale_amt
4  2018-03-15   West  Timothy  ...  56.0     1198.0   67088.0
8  2018-05-22   West  Douglas  ...  32.0     1198.0   38336.0
25 2019-03-07   West  Timothy  ...   7.0      500.0    3500.0
35 2019-08-24   West  Timothy  ...   3.0      125.0     375.0
37 2019-09-27   West  Timothy  ...  76.0      225.0   17100.0
38 2019-10-14   West  Douglas  ...  57.0      500.0   28500.0

'''

# to specify column by 'values' (only we get sale amnt column)
# print(pd.pivot_table(data, index=["Region", "Manager", "SalesMan"], values="Sale_amt"), '\n')
'''
O/P:
                              Sale_amt
Region  Manager SalesMan               
Central Douglas John       41338.666667
        Hermann Luis       41274.600000
                Shelli      8424.500000
                Sigal      41679.166667
        Martha  Steven     49922.500000
        Timothy David      28191.000000
East    Douglas Karen      16068.000000
        Martha  Alexander  29587.875000
                Diana      18050.000000
West    Douglas Michael    33418.000000
        Timothy Stephen    22015.750000 
'''

# Can use aggregate fn using "aggfunc" ('mean', 'sum', etc....)
# print(pd.pivot_table(data, index=["Region"],aggfunc="mean"))
'''
O/P:
        Sale_amt  Unit_price      Units
Region                                      
Central  34573.729167  645.458333  49.958333
East     24692.846154  442.038462  53.153846
West     25816.500000  624.333333  38.500000
'''

# Q) Find unit price of each sales man and their managers and region.
# df = pd.pivot_table(data, index=["Region", "Manager", "SalesMan"], values="Unit_price")
# print(df, '\n')

# Q) Find item wise average unit sold:
# df = pd.pivot_table(data, index="Item", values="Units")
# print(df, '\n')  # NB: By default the aggfunc will be mean of the values.

# Q) Find region wise total sale amount:
# df = pd.pivot_table(data, index="Region", values="Sale_amt", aggfunc="sum")
# print(df)

# Q) Count manager wise sale orders and total sale amount for each manager.
# df = pd.pivot_table(data, index="Manager", values="Sale_amt", aggfunc=["sum", len])
# print(df)
# df = pd.pivot_table(data, index=["Manager", "SalesMan"], values="Sale_amt", aggfunc=[np.sum])
# print(df)  # Margins=True is given for getting the total sale amount for the whole df

# df1 = df.query('Manager == ["Douglas"]')
# print(df1)

# Q) Find region wise no.of 'Television' and 'Home Theatres" sold
# df = pd.pivot_table(data, index=["Region", "Item"], values="Units", aggfunc="sum")
# df1 = df.query('Item == ["Television", "Home Theater"]')
# print(df1)

# Q) Find the max and min sale amt of the items:
# df = pd.pivot_table(data, index=["Item"], values="Sale_amt", aggfunc=["max", "min"])
# print(df)







