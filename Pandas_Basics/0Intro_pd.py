"""
Data Structures in pandas:
1) Series - 1D array (capable of holding data)
2) DataFrame - 2D labeled tabular structure with heterogeneously typed columns.
3) Panel - 3D labeled array.

NB: While giving o/p, pandas by default shows the data type of the elements of the array, the index/pos of each elements
    and the elements itself.
"""
import pandas as pd
import numpy as np

# Series can be created in following ways.

# 1) From a scalar value (a const value)

# s1 = pd.Series(10)
# print(s1)  # O/P will be : (index value) by default the index will be 0.
# print(type(s1))
#
# s1 = pd.Series(10, dtype='int32')  # Change data type
# print(s1)
""" Custom Indexing : By default, Pandas have the index pos starting from 0,1,2,3.... We can change or customize the 
index by passing the index pos as a list"""
# s2 = pd.Series(10, index=[1, 2, 3, 4])  # custom indexing
# print(s2)
# print(type(s2))
#
# s2 = pd.Series(10, index=['a', 'b', 'c'])  # custom indexing
# print(s2)
# print(type(s2))


'''CREATING A SERIES BY USING PANDAS FROM A LIST'''
# s1 = pd.Series([10, 20, 30, 'ah'])
# print(s1)
"""For s1 the data type will be 'object' since one of the element of array s1 have a string."""
# s2 = pd.Series([10, 20, 30, 40])
# print(s2)
"""For s2 the data type will be int64 since all the elements of the array are integers."""

'''CREATING A SERIES USING PANDAS.SERIES FROM A DICTIONARY'''
# d = {'x': 11, 'y': 22, 'z': 33}
# s1 = pd.Series(d)
# print(s1)

# From numpy 1D array
# import numpy as np
# a = np.array([4, 5, 6])
# s1 = pd.Series(a)
# print(s1)

# import numpy as np
# a = np.array([4, 5, 6])
# s1 = pd.Series(a, index=['a', 'b', 'c'])
# print(s1, '\n')
#
# print(s1[1], '\n')
# print(s1[0], '\n')
# print(s1[:], '\n')
# print(s1[-1], '\n')
# print(s1[:-1], '\n')
# print(s1[::2], '\n')
# print(s1['a'], '\n')

# s1 = pd.Series([10, 20, 30, 'h'], index=['a', 'b', 'c', 'd'])
# print(s1)
# print(s1.dtype)
# print(s1.size)
# print(s1.shape)
# # print(s1.data)
# print(s1.ndim)

'''SORTING'''
# s1 = pd.Series([10, 30, 15, 20], index=['a', 'b', 'c', 'd'])
# print(s1)
# We can sort index
# s = s1.sort_index()
# print(s)
# s = s1.sort_index(ascending=False)  # Sorting by descending order of index
# print(s)
# We can sort values
# s = s1.sort_values()
'''NB: When we sort the values of a series, the index will be considered the default index of values.
If we want to ignore indexing and assign new indexing for the sorted elements, we need to pass the 
arg ignore_index=True.'''
# s = s1.sort_values(ignore_index=True)
# print(s)

# serd = pd.Series(range(6), index=["white", "white", "blue", "green", "green", "yellow"])
#
# print(serd)
# print(type(serd))
# print(serd["white"])  # Will get a Series in place of a single element.

# To check any duplicate in index
# print(serd.index.is_unique)  # If duplicate index are present in the series, o/p will be false, else it will be true.


'''FILTERING'''
# Many ops applicable to NumPy arrays are extended to the Series.
# s = pd.Series([12, -4, 7, 9, 7], index=['a', 'b', 'c', 'd', 'e'])
# print(s)
# print(s > 8)  # O/P will be True or false for resp index
# print(s[s > 8])  # O/P will be the values > 8 with its index and datatype
# k = s[s > 8]
# print(k)

# Unique Values
# print(s.unique())  # Filters the duplicate data/values

# To count values
# print(s.value_counts())  # Gives the number of each elements present in the series.

# Evaluates the membership.
# print(s.isin([12, 7, 30]))  # O/P : True or False..
# print(s[s.isin([12, 7, 30])])

"""NaN (Not a Number) is used within pandas data structure to indicate the presence of an empty field
or not definable numerically"""
# import numpy as np
# s2 = pd.Series([5, -3, np.NaN, 14])
# print(s2)
"""is null() and notnull() fns are very useful to identify the indexes w/o a value."""
# print(s2.isnull())  # Finds the values which is NaN and the O/P will be true for that index and for others it will be
# #                     false.
# print(s2.notnull())  # True for index with values and False for index with NaN.
# print(s2[s2.notnull()])  # Filters the series with values from NaN
# print(s2[s2.isnull()])

'''FILTERING FROM A SERIES USING KEYS'''

# mydict = {'red': 2000, 'blue': 1000, 'yellow': 500, 'orange': 1000}
# myseries = pd.Series(mydict)
# print(myseries)

# colors = ['red', 'yellow', 'orange', 'blue', 'green']
'''The key values which are to be filtered out from the series is created as a list and passed as an argument(index=)'''
#
# myseries1 = pd.Series(myseries, index=colors)
# print(myseries1)
"""
NB: If the index list element is not there in the series (myseries), the corresponding value
to the index (here 'green') will be NaN. The order or index values will be taken from the series
which is passed through the args.
"""

# mydict1 = {'red': 2000, 'blue': 1000, 'yellow': 500, 'orange': 1000}
# myseries1 = pd.Series(mydict1)
# print(myseries1)
#
# mydict2 = {'red': 400, 'yellow': 1000, 'black': 700}
# myseries2 = pd.Series(mydict2)
# print(myseries2)
#
# print(myseries1 + myseries2)  # Common index values will be added all the other index will be NaN

# import numpy as np
# dict = {'red': 2000, 'blue': 1000, 'yellow': 500, 'orange': 1000, 'red1': 2000, 'blue1': 1000}
# s = pd.Series(dict)
#
# print(s.axes)
# print(s.dtype)
# print(s.ndim)
# print(s.size)
# print(s.values)
# print(s.head())  # Gives the first n-1 terms of the series from 0th pos(head).
# print(s.head(2))  # Gives the first (2-1) terms of the series from 0th pos(head).
# print(s.tail())  # Gives the first n-1 terms of series from las pos(tail).

"""
DATA FRAME : Data Frames  are used for creating heterogeneous data type (Tab structure)
NB: A series is a 1D array and a DataFrame is multidimensional array.
Used for heterogeneous data.
Size is mutable.
Data is also mutable.
It have labelled axis.
Can load data from diff format.
"""

# df1 = pd.DataFrame([10, 20, 30])
# print(df1)
#
# df2 = pd.DataFrame([10, 20, 30], ['a', 'b', 'c'])
# print(df2)
#
# df3 = pd.DataFrame([[10, 20, 30], [40, 50, 60]])
# print(df3)
#
# df4 = pd.DataFrame([[10, 20, 30], [40, 50, 60]], ['a', 'b'])  # custom index
# print(df4)
#
# df5 = pd.DataFrame([[10, 20, 30], [40, 50, 60]], ['a', 'b'], columns=['id', 'age', 'mark'])
# print(df5)
#
# df5.set_index('id', inplace=True)  # We can convert any particular column of df5 into index.
# print(df5)

# df6 = pd.DataFrame([[1, 'Ajith', 3], [2, 'Alex', 6]], columns=['ID', 'NAME', 'MARK'])
# print(df6)

'''USING DICT'''
'''NB: We cannot pass a 1D array for a DataFrame. 
       eg: If d = {'pid':1, 'name':'Abhijith', 'mark': 100}
           and if we use pd.DataFrame(d)  it generates an error as because DataFrame module of pandas is created for 
           handling multi dimensional arrays. 
'''
# d = {'pid': (1, 2, 3), 'name': ['book', 'pen', 'pencil'], 'price': [20, 10, 5]}
# df = pd.DataFrame(d)
# print(df)
#
# df.set_index('pid', inplace=True)  # making one column as index
# print(df)
#
# DataF = pd.DataFrame(d, index=['p1', 'p2', 'p3'])  # giving custom index to dict d.
# print(DataF)

# If we give a list of dict, it will group it up by key.
# If there is no value, it fill with NaN (Not a Number)
# l = [{'A': 'ann', 'B': 'tom'},
#      {'C': 'jerry', 'A': 'kevin'}]
# df = pd.DataFrame(l)
# print(df)


# l = [[1, 2, 3, 'Abhi'], [2, 5, 6, 'Sivu'], [3, 8, 9, 'Anju'], [4, 8, 9, 'Balu'], [5, 9, 8, 'Vivi'], [6, 5, 9, 'Supru']]
# df = pd.DataFrame(l, index=['a', 'b', 'c', 'd', 'e', 'f'], columns=['id', 'age', 'mark', 'name'])
# print(df)
# print(df.columns)  # To know column names
# print(df.keys())  # To get columns
# print(df.values)  # To get the values
# print(df.index)  # To get index range
# print(df.dtypes)  # To get column wise data types
# print(df.shape)  # To get the no. of rows and columns present in the DataFrame
# print(df.shape[0])  # To get the no. of columns of 0th row.
#
# print(df.describe())  # To see the summary of DataFrame.
# print(np.percentile(df['id'], 25))  # To get the value of 25th percentile of 'id' column of DataFrame.
# print(np.percentile(df['mark'], 25))  # To get the value of 25th percentile of 'mark' from df.
# print(np.percentile(df, 25))  # Error will be generated as because theres a string in a column(name)(CATEGORICAL DATA)
# #                               which cannot be quantified. So we need to mention the column.
# print(df.size)  # No. of elements in df
# print(df.nd)  # No of dimensions in df.

# l = [[1, 2, 3], [2, 5, 6], [3, np.nan, 9], [4, 8, 9], [5, 9, 8], [np.nan, 5, 9]]
# df = pd.DataFrame(l, index=['a', 'b', 'c', 'd', 'e', 'f'], columns=['id', 'age', 'mark'])
# print(df)
#
# print(df.nunique())  # By default it will be computing column wise
# print(df.nunique(axis=0))  # Column wise
# print(df.nunique(axis=1))  # jRow wise


# DATA SLICING

# l = [[1, 2, 3], [2, 5, 6], [3, 8, 9], [4, 8, 9], [5, 9, 8], [6, 8, 9]]
# df = pd.DataFrame(l, index=['a', 'b', 'c', 'd', 'e', 'f'], columns=['id', 'age', 'mark'])
# print(df)
# print(df['id'])  # to get specific column
# print(df.id)  # to get datatype of id column
# print(df.mark)  # to get data type of mark column
# print(df[['id', 'age']])  # to get specific columns from df

# To get specific rows
# print(df.loc['b'])  # used for custom index.
# print(df.iloc[1])  # used for actual/default row index. (i - 0,1,2,3....)
# print(df.loc[['b', 'f']])  # multiple rows
# print(df.iloc[[1, 3]])
# print(df.loc['b':'f'])
# print(df.iloc[1:3])

# print(df.loc[:, ['id', 'age']])  # Fetching the entire values of columns id and age.
# # OR
# print(df.iloc[:, [0, 1]])
# print(df.loc['b':, ['id', 'age']])  # To fetch all elements of column id and age for rows from b to the last.
# print(df.loc[['b', 'e'], ['id', 'age']])  # To fetch elements of column id and age for rows b and e using loc.
# print(df.iloc[[1, 4], [0, 1]])  # To fetch elements of column id and age for rows b and e using iloc.

# print(df.iloc[-1, -1])
# print(df.iloc[-1:, -1:])
# print(df.iloc[-3:, -2:])

# To achieve a single value within a dataframe, first you have to the name of the column and then the index or the
# label of the row.

# print(df['id']['b'])  # [column][row]

# Methods to index: to get some info about index from a data structure.
# print(df.iloc[:, 1].idxmax())
# print(df.iloc[:, 1].idxmin())


# CONDITIONAL SLICING
# l=[[1,2,3,'ajith','mech'],[2,5,6,'alex','mech'],[3,8,9,'john','cs'],[4,8,9,'ann','ec'],[5,9,8,'smith','cs']]
# df=pd.DataFrame(l,index=['a','b','c','d','e'],columns=['id','age','mark','name','dept'])
# print(df, '\n')

# print(df['age']>5)  # O/P will be boolean
# print(df[df['age']>5])  # O/P will be a dataframe
# print(df[df['mark']==9])
# OR
# print(df[df.mark == 9])
# OR
# print(df.loc[df['mark']==9])
# OR
# print(df.loc[df.mark==9])
'''iloc can not be used since the default statement for conditional slicing will be considering the custom index.'''

# For slicing df with .loc we can do it by mentioning the column name as well.
# print(df.loc[df.mark>6, 'mark'], '\n')
# print(df.loc[df.mark>6, ['mark', 'age']], '\n')
# df1 = df.loc[df.mark>6, 'mark']
# print(df1, '\n')
# df2 = df.loc[df.mark>6, ['mark', 'age']]
# print(df2, '\n')
# df3 = df[df['mark']>6]  # We cannot mention the column name along with this.
# print(df3)

# Two conditions using logical operator
# We can use &
# df1 = df[(df['mark']<=6) & (df['dept']=='mech')]
# print(df1)
# df1 = df.loc[(df['mark']<=6) & (df['dept']=='mech'), 'id']
# print(df1)

# Q) Mark b/w 7 and 9
# df1 = df.loc[(df['mark']>7) & (df['mark']<9)]
# print(df1)
# If we want to use iloc.
# df1 = df[df.iloc[:,2]==9]
# print(df1)

# OR operator |
# print(df[(df.mark<=3) | (df.dept=='mech')])  # For OR(|) cond anyone cond is applicable.
# df1 = df[df.dept.isin(['cs', 'mech', 'ec'])]
# print(df1)
# df1 = df[df.dept.isin(['cs', 'mech']) & (df.mark<9)]
# print(df1)

# Updating the dataframe

# df['age']+=2
# print(df)  # All the elements of age will be incremented by 2
# df.loc['b', 'age']+=2
# print(df)

# Assign values to a dataframe if a condition satisfies:

# df.loc[df.mark>6, 'mark']=100  # Replacing all the marks > 6 with 100
# print(df)

# DataFrame operations:
# l=[[1,2,3],[2,5,6],[3,8,9],[4,8,9],[5,9,8],[6,8,9]]
# df=pd.DataFrame(l,index=['a','b','c','d','e','f'],columns=['id','age','mark'])
# print(df, '\n')

# To add new rows
# df2 = pd.DataFrame([[7, 10, 6], [8, 12, 8]],index=['g','h'],columns=['id', 'age', 'mark'])
# print(df2, '\n')
# df = df.append(df2)
# print(df, '\n')

# To delete row
# df = df.drop(['a', 'b'])
# print(df)

# To add new column with all values same
# df['new'] = 12
# print(df)

# df['new'] = df['age']+df['mark']
# print(df)

# To add new column with diff values
# df['new'] = [1, 2, 3, 4, 5, 6]  # The no.of rows should match with the list size
# print(df)

# To delete column by using drop:
# df = df.drop(['new'], axis=1)  # By default it will be deleting a row but for column we should specify axis=1
# print(df, '\n')
'''NB: We cannot use the drop fn to the existing dataframe, instead we need to assign a new variable and create 
a new dataframe.'''

# DataFrame membership, same as in series
# print(df.isin([2, 4]), '\n')  # Boolean
# print(df[df.isin([2, 4])], '\n')  # Float

# Create a dictionary of series:
# d = {'Name':pd.Series(['Tom','James','Ricky','Vin','Steve','Smith','Jack']),
#    'Age':pd.Series([25,26,25,23,30,29,23]),
#    'Rating':pd.Series([4.23,3.24,3.98,2.56,3.20,4.6,3.8])}
# df = pd.DataFrame(d)
# print('Our DataFrame is:')
# print(df, '\n')
# print(df.shape, '\n')
# # print(df.T, '\n')  # Transpose of DataFrame
# dfnew = df.T  # Creates a new dataframe of Transpose of df
# print(dfnew, '\n')
# print(dfnew.shape)

'''
                       STATISTICS:
Statistics are of 2 types:
1) Descriptive: Descriptive statistics are used for summarization of data.
2) Inferential: Inferential statistics are used when data is viewed as a subclass of a specific population.
eg: Similar to that of classical and statistical Thermodynamics.
'''
# d = {
#    'Age':pd.Series([25,26,25,23,30,29,23]),
#    'Rating':pd.Series([4.23,3.24,3.98,2.56,3.20,4.6,3.8])}
# df = pd.DataFrame(d)
# print(df, '\n')
# print(df.sum(), '\n')  # default is axis=0, column wise
# print(df.sum(1), '\n')  # row wise

# To get avg value
# print(df.mean(), '\n')  # default axis=0, column wise

'''
Variance: avg square distance from mean.
Standard Deviation: sqr root of variance.
ie; avg distance of points from mean value.
(If we know spread, we get an idea that how wide our points spread from mean, is it around
mean value or not.
'''

# To get standard deviation:
# print(df.std(), '\n')
# print(df.median(), '\n')  # Median is the central tendency of a dataframe
# print(df.mode(), '\n')  # Mode is the most repeated data of a dataframe
# print(df.max())
# print(df.min())
# print(df.cumsum())

# There are 3 types of


'''To every element in the table/to the whole data frame we can perform '''

# def adder(ele1, ele2):
#     print(ele1)
#     print(ele2)
#     return ele1+ele2
#
# df = pd.DataFrame(np.random.randn(5, 3), columns=['col1', 'col2', 'col3'])
# print(df)
# print(df.pipe(adder, 2))  # pipe passing df and 2 as args and call the fn

'''Arbitrary fns can be applied along the axes of a DataFrame using the apply() method'''
# We will get single values per column/row

# print(df.apply(np.mean))
# print(df.apply(np.mean, axis=1))
# print(df.apply(lambda x: x.max() - x.min()))
# print(df.apply(lambda x: x.max() - x.min(), axis=1))

# Element wise fn application (map, applymap)
# we can perform operation each elements

# My custom fn
# print(df['col1'].map(lambda x: x*100))  # for single column (or we can say a series)
# print(df.col1.map(lambda x: x*100))  # same as above
# print(df.loc[1].map(lambda x: x*100))  # for single row or we can say a series
# print(df.applymap(lambda x: x*100))  # for all dataframe elements

# d = {'Name': pd.Series(['Tom', 'James', 'Ricky', 'Vin', 'Steve', 'Smith', 'Jack']),
#      'Age': pd.Series([25, 26, 25, 23, 30, 29, 23])}
# df = pd.DataFrame(d)
# print(df, '\n')
# # Finding name that starts with 'S'
# print(df.Name.map(lambda x: x.startswith('S')), '\n')  # Boolean o/p
# print(df[df.Name.map(lambda x: x.startswith('S'))], '\n')  # dataframe o/p
# print(df.loc[df.Name.map(lambda x:x.startswith("S"))])  # Same as above

# Reindex like (it makes df index similar to matching df)
# fill with NaN or remove rows if any mismatch
# df1 = pd.DataFrame(np.random.randn(10, 4), columns=['col1', 'col2', 'col3', 'col4'])
# print(df1, '\n')
# df2 = pd.DataFrame(np.random.randn(7, 3), columns=['col1', 'col2', 'col3'])
# print(df2, '\n')
# df1 = df1.reindex_like(df2)
# print(df1, '\n')  # If the number of rows or columns of child dataframe is greater than the parent dataframe, then for
#                     reindexing the row and column will be removed from the child dataframe.
# df2 = df2.reindex_like(df1)
# print(df2)  # If the number of rows or columns of child df is less than the parent df then the extra rows or columns
#             will be filled with NaN which is obsolete.
# Hence for filling with other elements other than NaN.
# Filling while reindexing
# print(df2.reindex_like(df1, method='ffill'))  # fill with last row

# If the df is custom labeled, it replace with another data frame index pos and name.
# df=pd.DataFrame({"name":["ann","anu","jiss"],"age":[15,48,26],"marks":[45,78,25]},index=["D","E","A"])
# print(df, '\n')
# df1=pd.DataFrame({"name":["anju","kiran","amy"],
#                   "age":[15,14,19],
#                   "marks":[45,85,62]},index=["D","B","A"])
# print(df1, '\n')
# data = df.reindex_like(df1)
# print(data, '\n')
#
# # df = df.reindex(['A', 'E', 'C', 'B'])  # if the given index is not there already, then
# # print(df)
#
# df = df.reindex(columns=['age', 'marks', 'name'])  # if we want to change the pos of columns.
# print(df)

# To set date dynamically as index or something like this, We can set datedata as index
# datedata=pd.date_range(start="10/1/2019",end="10/3/2019")
# print(datedata)
# df2=pd.DataFrame({"name":["anju","kiran","amy"],
#                   "age":[15,14,19],
#                   "marks":[45,85,62]},index=datedata)
# print(df2)

# Rename
# df1 = pd.DataFrame(np.random.randn(6, 3), columns=['col1', 'col2', 'col3'])
# print(df1)
# print('After renaming the rows and columns:')
# print(df1.rename(columns={'col1': 'c1', 'col2': 'c2'},
#                 index={0: 'apple', 1: 'banana', 2: 'durian'}))

# Drop
# For dropping rows: By default pandas will drop columns or you can set the axis as 0.
# dframe = pd.DataFrame(np.arange(16).reshape(4, 4), index=['red', 'blue', 'yellow', 'white'],
#                       columns=['ball', 'pen', 'pencil', 'paper'])
# print(dframe)
# print(dframe.drop(['red']))  # NB: dframe.drop will not change or deletes the row of df dframe instead it allocates to
#                                  a new dataframe.
# print(dframe)  # When we print the dframe it will be same as that of initial one.
# print(dframe.drop(['red', 'blue']))  # List of rows for removing multiple rows

# For dropping columns: Set the axis as 1.
# print(dframe.drop(['ball', 'pencil'], axis=1))

# DataFrame Iteration:
# d = {'pid': [1, 2, 3], 'name':['book', 'pen', 'pencil'], 'price': [20, 10, 5]}
# df = pd.DataFrame(d)
# print(df, '\n')

'''
iteritems() - to iterate over the (key, value) pairs
iterrows() - to iterate over the rows as (index, series) pairs
itertuples() - iterate over the rows as namedtuples.
'''
# for key,value in df.iteritems():
#     print(key)
#     print(value)
# print('\n')
# for x in df.iteritems(): # iterating row wise
#     print(x[1])
# for i in df.iterrows():
#     print(i)
# for row_index, row in df.iterrows():
#     print(row_index)
#     print(row)

# for row in df.itertuples():
#     print(row)
# print('\n')
# for row in df.itertuples():
#     print(row[1])


# Sorting of DataFrame:
'''There are 2 kinds of sorting for DataFrame available in pandas. They are:-
   1) By label
   2) By values
'''
# l = [[1, 2, 3], [2, 5, 6], [3, 8, 9], [4, 8, 9], [5, 9, 8], [6, 8, 9]]
# df = pd.DataFrame(l, index=['a', 'c', 'd', 'e', 'b', 'f'], columns=['id', 'age', 'mark'])
# print(df, '\n')

# we can sort by values, get new data frame
# print(df.sort_values(by='age'))  # must specify by column name
# print(df.sort_values(by=['age', 'mark']))  # we will not get correct sorting
# dfnew = df.sort_values(by='age', ascending=False)
# print(dfnew)


# Merge sort
# sort_values() provides to choose the algorithm 'mergesort'

# unsorted_df = pd.DataFrame({'col1':[2,1,1,1],'col2':[1,3,2,4]})
# print(unsorted_df)
# sorted_df = unsorted_df.sort_values(by='col1',kind='mergesort')
# print(sorted_df, '\n')

# DataFrame Text data:
# String ops for a series:
# s = pd.Series(['Tom', 'William Rick', 'John', 'Alber@t', np.nan, '1234', 'SteveSmith'])
# print(s)
# print(s.str.lower())
# print(s.str.upper())
# print(s.str.len())
# print(s.str.strip())
# print(s.str.len())
# s1 = s.str.strip()
# print(s1.str.len())
# print(s.str.cat(sep='/'))  # to join using a separator.

"""
                    LABEL ENCODING:
=> We have label encoding and one hot encoding for strings.
=> In label encoding, we gives 1, 2, 3... for unique string.
   (In LE, machine may find out any relation between digits, like 1+2=3, 1<3, etc... So we follows OHE.
=> In OHE, we follow binary numbers.
   (It forms a vector, its length = no. of unique strings)
   If the number of categories increases, vector become sparse.

designation               teacher   student   office-staff
 teacher                     1         0            0
 teacher                     1         0            0
 student                     0         1            0
 student                     0         1            0
 student                     0         1            0
 office-staff                0         0            1
 office-staff                0         0            1 # This extra variables are called dummy variables.
"""

# print(s.str.get_dummies())  # To get one hand encoding
# OR
# print(pd.get_dummies(s))
# print(s.str.replace('@', '$'), '\n')
# print(s.str.startswith('T'), '\n')  # Boolean o/p
# print(s.str.endswith('t'), '\n')
# print(s.str.find('e'), '\n')  # -1 if not matching
# print(s.str.findall('e'), '\n')  # returns a list of e's with no.of repetitions in each string
# print(s.str.swapcase(), '\n')  # replaces the lower with upper cases and vice versa.
# print(s.str.islower(), '\n')  # Boolean
# print(s.str.isupper(), '\n')
# print(s.str.isnumeric(), '\n')


# String ops for a DataFrame:
# d = {'Name':pd.Series(['Tom','James','Ricky','Vin','Steve','Smith','Jack']),
#    'Age':pd.Series([25,26,25,23,30,29,23]),
#      'Sex':pd.Series(['male','female','male','male','male','male','female']),
#    'Rating':pd.Series([4.23,3.24,3.98,2.56,3.20,4.6,3.8])}
# df = pd.DataFrame(d)
# print(df, '\n')
# print(df['Name'].str.lower(), '\n')
# print(df['Name'].str.len(), '\n')
# print(df['Name'].str.get_dummies(), '\n')
# print(pd.get_dummies(df), '\n')
# df['Name'] = df['Name'].str.lower()
# print(df, '\n')
# df.set_index('Name', inplace=True)
# print(df)

# Arithmetic ops:
# df1 = pd.DataFrame(np.array([1, 2, 3, 4]).reshape(2, 2), index=[1,2], columns=[1, 2])
# df2 = pd.DataFrame(np.array([7, 8, 5, 6]).reshape(2, 2), index=[1, 2], columns=[1, 2])
# print(df1, '\n')
# print(df2, '\n')
# print(df1+df2, '\n')
# # If the index doesn't match then the elements will be NaN , it will be filled with NaN
# df3 = pd.DataFrame(np.array([8, 9, 7, 6]).reshape(2, 2), index=[1, 3], columns=[1, 2])
# print(df3+df1)
#

# Merging of DataFrames:
# Merging DataFrames using id


# df1 = pd.DataFrame({
#    'id':[1,2,3,4,5],
#    'Name': ['Alex', 'Amy', 'Allen', 'Alice', 'Ayoung'],
#    'subject_id':['sub1','sub2','sub4','sub6','sub5']})
# print(df1, '\n')
#
# df2 = pd.DataFrame(
#    {'id':[1,2,3,4,5],
#    'Name': ['Billy', 'Brian', 'Bran', 'Bryce', 'Betty'],
#    'subject_id':['sub2','sub4','sub3','sub6','sub5']})
# print(df2, '\n')

'''merging using how (how can have 4 values - left, right, inner, outer)'''
# inner join:
# print(pd.merge(df1, df2, on='subject_id'), '\n')
# lft
# print(pd.merge(df1, df2, on='subject_id', how='left'), '\n')
# right
# print(pd.merge(df1, df2, on='subject_id', how='right'), '\n')
# outer join
# print(pd.merge(df1, df2, on='subject_id', how='outer'), '\n')

'''
                   MELTING AND PIVOT
                   -----------------
Melting: Melting in pandas reshape data frame from wide format to long format.
It reduces the columns and increases the rows of a DataFrame.
'''

# df = pd.DataFrame(data={
#     'Day': ['MON', 'TUE', 'WED', 'THU', 'FRI'],
#     'Google': [1129, 1132, 1134, 1152, 1152],
#     'Apple': [191, 192, 190, 190, 188],
#     'Amazon': [210, 524, 430, 325, 551]
# })
# print(df, '\n')
'''
Here we can use 'day' as a variable/identifier because it is common to other fields.
so on 'day' basis we can do reshape.
that we want to mention as "id_vars=["col_names"]"
'''
# df1 = df.melt(id_vars=['Day'])
# print(df1, '\n')
'''The above example will melt the Day column into rows, but the values of the dataframe 
will be undefined and hence it will be designated as 'values' only and the variables 
(ie; company) as variables only
   Day variable  value
0   MON   Google   1129
1   TUE   Google   1132
2   WED   Google   1134
3   THU   Google   1152
4   FRI   Google   1152
5   MON    Apple    191
6   TUE    Apple    192
7   WED    Apple    190
8   THU    Apple    190
9   FRI    Apple    188
10  MON   Amazon    210
11  TUE   Amazon    524
12  WED   Amazon    430
13  THU   Amazon    325
14  FRI   Amazon    551

'''
# df2 = df.melt(id_vars=['Day'], value_vars=['Google', 'Amazon'], var_name='company', value_name='sales')
# print(df2, '\n')
'''In this example, we melt the Day column of DataFrame into rows and assign the value_name
as 'sales' and variable names as 'Company'.
Also, in this example we have chosen the variables for which the data to be melted.
ie; "value_vars=['column_name1','column_name2']" 
  Day company  sales
0  MON  Google   1129
1  TUE  Google   1132
2  WED  Google   1134
3  THU  Google   1152
4  FRI  Google   1152
5  MON  Amazon    210
6  TUE  Amazon    524
7  WED  Amazon    430
8  THU  Amazon    325
9  FRI  Amazon    551 
'''

'''Pivot: Reverse of Melt (opp. concept of melting)'''
# originaldata = df2.pivot(index='Day', columns="company")["sales"].reset_index()
# originaldata.columns.name = None
# print(originaldata)

'''
                             CONCATENATION:
                             -------------
Pandas provides various facilities for easily combining together series, DataFrame, and panel 
objects.
'''


# one = pd.DataFrame({'Name': ['Alex', 'Amy', 'Allen', 'Alice', 'Ayoung'],
#                     'subject_id': ['sub1', 'sub2', 'sub4', 'sub6', 'sub5'],
#                     'Marks_scored': [98, 90, 87, 69, 78]
#                     }, index=[1, 2, 3, 4, 5])
#
# two = pd.DataFrame({
#     'Name': ['Billy', 'Brian', 'Bran', 'Bryce', 'Betty'],
#     'subject_id': ['sub2', 'sub4', 'sub3', 'sub6', 'sub5'],
#     'Marks_scored': [89, 80, 79, 97, 88]},
#     index=[1, 2, 3, 4, 5])
# print(one, '\n')
# print(two, '\n')
#
# print(pd.concat([one, two]), '\n')
'''
Combine with df's index.
One below the other df model.
    Name subject_id  Marks_scored
1    Alex       sub1            98
2     Amy       sub2            90
3   Allen       sub4            87
4   Alice       sub6            69
5  Ayoung       sub5            78
1   Billy       sub2            89
2   Brian       sub4            80
3    Bran       sub3            79
4   Bryce       sub6            97
5   Betty       sub5            88

'''

# print(pd.concat([one, two], ignore_index=True), '\n')
'''
Default index:
Here the index annotation will be ignored, even though the index pos will not be ignored
while concatenating. Default index will be assigned to the concatenated DF.
    Name subject_id  Marks_scored
0    Alex       sub1            98
1     Amy       sub2            90
2   Allen       sub4            87
3   Alice       sub6            69
4  Ayoung       sub5            78
5   Billy       sub2            89
6   Brian       sub4            80
7    Bran       sub3            79
8   Bryce       sub6            97
9   Betty       sub5            88 

'''

# print(pd.concat(objs=[one, two], axis=0), '\n')  # under same columns. (By default)
# print(pd.concat(objs=[one, two], axis=1), '\n')  # under same row.

'''
                             NaN HANDLING:
                             ------------
NaN is created using numpy(numpy.nan)
'''
# df = pd.DataFrame([[1, 3, 2], [np.nan, 2, 5], [5, 7, 10], [4, 8, 3], [5, np.nan, 6]], columns=['c1', 'c2', 'c3'])
# print(df, '\n')
'''
c1   c2  c3
0  1.0  3.0   2
1  NaN  2.0   5
2  5.0  7.0  10
3  4.0  8.0   3
4  5.0  NaN   6 
'''
# print(df.isna(), '\n')
'''
To check whether NaN values are present in the whole DF.
   c1     c2     c3
0  False  False  False
1   True  False  False
2  False  False  False
3  False  False  False
4  False   True  False 
'''
# print(df.loc[4, :].isna(), '\n')
'''
To check whether the 4th row of DF have NaN value in any columns.
c1    False
c2     True
c3    False
Name: 4, dtype: bool 
'''
# print(df.dropna(), '\n')
'''
To drop row with value NaN and create a new dataframe from the existing DF.
 c1   c2  c3
0  1.0  3.0   2
2  5.0  7.0  10
3  4.0  8.0   3 
'''
# dfnew = df.fillna(10)
# print(dfnew, '\n')
'''
# to fill NaN with the value 10 and create a new dataframe.
 c1    c2  c3
0   1.0   3.0   2
1  10.0   2.0   5
2   5.0   7.0  10
3   4.0   8.0   3
4   5.0  10.0   6 
'''
# dfnew = df.ffill()
# print(dfnew, '\n')
'''
To fill NaN with upper row value.
 c1   c2  c3
0  1.0  3.0   2
1  1.0  2.0   5
2  5.0  7.0  10
3  4.0  8.0   3
4  5.0  8.0   6 
'''
# dfnew  = df.bfill()
# print(dfnew, '\n')
'''
To fill NaN with below row.
  c1   c2  c3
0  1.0  3.0   2
1  5.0  2.0   5
2  5.0  7.0  10
3  4.0  8.0   3
4  5.0  NaN   6 

'''
# print(df.interpolate(), '\n')
'''
To fill NaN with any random value.
 c1   c2  c3
0  1.0  3.0   2
1  3.0  2.0   5
2  5.0  7.0  10
3  4.0  8.0   3
4  5.0  8.0   6 
'''
# dfnew = df.replace(2, np.nan)  # it form a new df, but we will get all data in df
# print(dfnew)

# Q) Replace 5 with NaN in the column 'c1' of df.
# dfnew = df.loc[:, 'c1'].replace(5, np.nan)
# print(dfnew)

# Q) Change NaN in c2 column with avg of c2.
# avg = df['c2'].mean()
# print(avg)
# dfnew = df['c2'].replace(np.nan, avg)
# print(dfnew)
# # OR
# dfnew = df.c2.fillna(df.c2.mean())  # new data frame
# print(dfnew)
'''NB: Inorder to increase the data accuracy by filling NaN, it is always recommended to
fill NaN with the avg value of particular column.'''

'''
                            DATE TIME FORMAT
                            ----------------
'''

df = pd.DataFrame({'specious': ['A', 'B', 'C'], 'days': [10, 12, 13],
                   'time': ['10/09/2010 2:30:0', '12/09/2010 4:30:0', '09/09/2010 5:30:0']})
print(df, '\n')

print(type(df.time[0]))  # data in str format
df['time'] = pd.to_datetime(df.time)
print(type(df.time[0]))  # data in datetime format.

# To get year only from that column dt.year
print(df.time.dt.year)
print(df.time.dt.weekday)  # sun = 0, mon = 1, .....
# #converting to time stamp
# print(type(df.time[0]))
# print(df)
# # # # to get year only from that column dt.year
# print(df.time.dt.year)
# print(df.time.dt.weekday)  #sun=0, mon=1...
# print(df.time.dt.day_name())
# print(df.time.dt.month)





