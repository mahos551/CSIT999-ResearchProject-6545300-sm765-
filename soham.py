#!/usr/bin/env python
# coding: utf-8

# In[1]:


#!pip install mlxtend
#!pip install tabula
#!pip install apriori_python
#!pip install PySimpleGUI
import pandas as pd
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules
import tabula
from apriori_python import apriori
from apriori_python.apriori import * 
from apriori_python.utils import *
from string import ascii_lowercase
import PySimpleGUI as sg
import re
import os.path
import PIL.Image
import io
import base64
from string import ascii_lowercase
import itertools


# In[2]:


#functions and all other process definitions
    
def powerset(s):
    return chain.from_iterable(combinations(s, r) for r in range(1, len(s)))

def associationRule(freqItemSet, itemSetWithSup, minConf):
    rules = []
    for k, itemSet in freqItemSet.items():
        for item in itemSet:
            subsets = powerset(item)
            for s in subsets:
                confidence = float(
                    itemSetWithSup[item] / itemSetWithSup[frozenset(s)])
                if(confidence > minConf):
                    rules.append([set(s), set(item.difference(s)), confidence])
    return rules

def getItemSetFromList(itemSetList):
    tempItemSet = set()

    for itemSet in itemSetList:
        for item in itemSet:
            tempItemSet.add(frozenset([item]))

    return tempItemSet

def apriori(itemSetList, minSup, minConf):
    C1ItemSet = getItemSetFromList(itemSetList)
    # Final result, global frequent itemset
    globalFreqItemSet = dict()
    # Storing global itemset with support count
    globalItemSetWithSup = defaultdict(int)

    L1ItemSet = getAboveMinSup(C1ItemSet, itemSetList, minSup, globalItemSetWithSup)
    currentLSet = L1ItemSet
    k = 2

    # Calculating frequent item set
    while(currentLSet):
        # Storing frequent itemset
        globalFreqItemSet[k-1] = currentLSet
        # Self-joining Lk
        candidateSet = getUnion(currentLSet, k)
        # Perform subset testing and remove pruned supersets
        candidateSet = pruning(candidateSet, currentLSet, k-1)
        # Scanning itemSet for counting support
        currentLSet = getAboveMinSup(candidateSet, itemSetList, minSup, globalItemSetWithSup)
        #print(currentLSet)
        k += 1

    rules = associationRule(globalFreqItemSet, globalItemSetWithSup, minConf)
    rules.sort(key=lambda x: x[2])

    return globalFreqItemSet, rules

def getUnion(itemSet, length):
    return set([i.union(j) for i in itemSet for j in itemSet if len(i.union(j)) == length])

def pruning(candidateSet, prevFreqSet, length):
    tempCandidateSet = candidateSet.copy()
    for item in candidateSet:
        subsets = combinations(item, length)
        for subset in subsets:
            # if the subset is not in previous K-frequent get, then remove the set
            if(frozenset(subset) not in prevFreqSet):
                tempCandidateSet.remove(item)
                break
    return tempCandidateSet

def getAboveMinSup(itemSet, itemSetList, minSup, globalItemSetWithSup):
    freqItemSet = set()
    #support_list = []
    localItemSetWithSup = defaultdict(int)

    for item in itemSet:
        for itemSet in itemSetList:
            if item.issubset(itemSet):
                globalItemSetWithSup[item] += 1
                localItemSetWithSup[item] += 1

    for item, supCount in localItemSetWithSup.items():
        support = float(supCount / len(itemSetList))
        if(support >= minSup):
            freqItemSet.add(item)
            #support_list.append(support)
            

    return freqItemSet


def convert_to_bytes(file_or_bytes, resize=None):
    '''
    Will convert into bytes and optionally resize an image that is a file or a base64 bytes object.
    Turns into  PNG format in the process so that can be displayed by tkinter
    :param file_or_bytes: either a string filename or a bytes base64 image object
    :type file_or_bytes:  (Union[str, bytes])
    :param resize:  optional new size
    :type resize: (Tuple[int, int] or None)
    :return: (bytes) a byte-string object
    :rtype: (bytes)
    '''
    if isinstance(file_or_bytes, str):
        img = PIL.Image.open(file_or_bytes)
    else:
        try:
            img = PIL.Image.open(io.BytesIO(base64.b64decode(file_or_bytes)))
        except Exception as e:
            dataBytesIO = io.BytesIO(file_or_bytes)
            img = PIL.Image.open(dataBytesIO)

    cur_width, cur_height = img.size
    if resize:
        new_width, new_height = resize
        scale = min(new_height/cur_height, new_width/cur_width)
        img = img.resize((int(cur_width*scale), int(cur_height*scale)), PIL.Image.ANTIALIAS)
    bio = io.BytesIO()
    img.save(bio, format="PNG")
    del img
    return bio.getvalue()


# In[3]:



def file_open( filename, conditions, df=None, df1=None ):
    df = pd.read_excel(filename)
    df1 = pd.read_excel(conditions)
    df = df.dropna(axis=1,how='all')
    df = df.dropna(axis=0, how = 'all')
    df = df.drop(["Patientid"],axis=1)
    df1 = df1.drop(["Patientid"],axis=1)
    df2 = pd.merge(df1, df, on="Time",how="outer")
    #df2 = df2[:50]
    df2 = df2.fillna(method='ffill')
    for i in range(len(df2["Conditions"])): 
        df2["Conditions"][i] = re.sub(r'[^\w]', ' ', df2["Conditions"][i])
        df2["Conditions"][i] = re.sub(r'patient[0-9]', ' ', df2["Conditions"][i])
        df2["Conditions"][i] = re.sub(r'normal | high | low | immobilize | adequate', ' ', df2["Conditions"][i])
    return df2


# In[4]:


# def powerset(s):
#     return chain.from_iterable(combinations(s, r) for r in range(1, len(s)))


# In[5]:


# def associationRule(freqItemSet, itemSetWithSup, minConf):
#     rules = []
#     for k, itemSet in freqItemSet.items():
#         for item in itemSet:
#             subsets = powerset(item)
#             for s in subsets:
#                 confidence = float(
#                     itemSetWithSup[item] / itemSetWithSup[frozenset(s)])
#                 if(confidence > minConf):
#                     rules.append([set(s), set(item.difference(s)), confidence])
#     return rules


# In[6]:


# def getItemSetFromList(itemSetList):
#     tempItemSet = set()

#     for itemSet in itemSetList:
#         for item in itemSet:
#             tempItemSet.add(frozenset([item]))

#     return tempItemSet


# In[7]:


# def apriori(itemSetList, minSup, minConf):
#     C1ItemSet = getItemSetFromList(itemSetList)
#     # Final result, global frequent itemset
#     globalFreqItemSet = dict()
#     # Storing global itemset with support count
#     globalItemSetWithSup = defaultdict(int)

#     L1ItemSet = getAboveMinSup(C1ItemSet, itemSetList, minSup, globalItemSetWithSup)
#     currentLSet = L1ItemSet
#     k = 2

#     # Calculating frequent item set
#     while(currentLSet):
#         # Storing frequent itemset
#         globalFreqItemSet[k-1] = currentLSet
#         # Self-joining Lk
#         candidateSet = getUnion(currentLSet, k)
#         # Perform subset testing and remove pruned supersets
#         candidateSet = pruning(candidateSet, currentLSet, k-1)
#         # Scanning itemSet for counting support
#         currentLSet = getAboveMinSup(candidateSet, itemSetList, minSup, globalItemSetWithSup)
#         #print(currentLSet)
#         k += 1

#     rules = associationRule(globalFreqItemSet, globalItemSetWithSup, minConf)
#     rules.sort(key=lambda x: x[2])

#     return globalFreqItemSet, rules


# In[8]:


# def getUnion(itemSet, length):
#     return set([i.union(j) for i in itemSet for j in itemSet if len(i.union(j)) == length])


# In[9]:


# def pruning(candidateSet, prevFreqSet, length):
#     tempCandidateSet = candidateSet.copy()
#     for item in candidateSet:
#         subsets = combinations(item, length)
#         for subset in subsets:
#             # if the subset is not in previous K-frequent get, then remove the set
#             if(frozenset(subset) not in prevFreqSet):
#                 tempCandidateSet.remove(item)
#                 break
#     return tempCandidateSet


# In[10]:


# def getAboveMinSup(itemSet, itemSetList, minSup, globalItemSetWithSup):
#     freqItemSet = set()
#     #support_list = []
#     localItemSetWithSup = defaultdict(int)

#     for item in itemSet:
#         for itemSet in itemSetList:
#             if item.issubset(itemSet):
#                 globalItemSetWithSup[item] += 1
#                 localItemSetWithSup[item] += 1

#     for item, supCount in localItemSetWithSup.items():
#         support = float(supCount / len(itemSetList))
#         if(support >= minSup):
#             freqItemSet.add(item)
#             #support_list.append(support)
            

#     return freqItemSet


# In[11]:


def fetch_itemsets(df2=None):
    itemSetList = df2.values.tolist()
    freqItemSet, rules = apriori(itemSetList,0.01,0.5)
    return itemSetList, freqItemSet, rules

#itemSetList,freqItemSet,rules= itemsets(df2)


# In[12]:


def itemsdict(df2=None):
    values = list(df2['Treatment'].unique())
    itemsets = {}
    pattern = re.compile(r'\s+')
    for value in values:
        itemsets[value] = []
        for i in range(len(df2)):
            if df2['Treatment'][i] == value:
                sentence = re.sub(' +', ' ',df2['Conditions'][i])
                itemsets[value].append(sentence)
    itemsets[values[0]] = list(set(itemsets[values[0]]))
    itemsets[values[1]]= list(set(itemsets[values[1]]))
    return itemsets


#itemsets = itemsdict(df2)    # -------- (1)


# In[13]:


def CMrules(rules,itemSetList):
    lhs_list = []
    rhs_list = []
    for i,value in enumerate(rules):
        lhs_list.append(list(value[0]))
        rhs_list.append(list(value[1]))
    strings = []
    for j,k in zip(lhs_list,rhs_list):
        var = ''.join(j)
        var1 = ''.join(k)
        strings.append(var+var1)
    Seq_strings = []
    for i in itemSetList:
        Seq_strings.append(''.join(i))

    for i,j,l in zip(rules,strings,lhs_list):
        seq_support_count = 0
        lhs_support_count = 0
        for k in Seq_strings:
            if j in k:
                seq_support_count += 1
        for l in lhs_list:
            l = ''.join(l)
            if l in k:
                lhs_support_count += 1

        seq_support = seq_support_count/len(Seq_strings)
        lhs_support = lhs_support_count/len(Seq_strings)        
        seq_confidence = seq_support/lhs_support
        i.append(seq_support)
        i.append(seq_confidence)
    return rules 


# In[14]:



def table2_obj(itemsets):
    letter_dict = dict()

    n_itemset = []

    for i in itemsets.values():
        n_itemset.extend(i)

    j = 0;    
    for i in set(n_itemset):
        letter_dict[i] =  "".join([x for x in itertools.product(ascii_lowercase, repeat=int(j/25)+1)][j%25])
        j = j + 1
    data = []
    for i in itemsets.values():
        data.append( " , ".join(sorted(set([letter_dict[x] for x in i]))))
    return [data]


# In[15]:



def t1_tweak(itemsets):
    max_ = 0
    for i in itemsets.values():
        max_ = len(i) if max_ < len(i) else max_

    temp = [[] for i in range(max_)]    
    
    for i in range(len(itemsets)): 
        for j in range(max_):
            val = " --- "
            if j < len(list(itemsets.values())[i]):
                val = list(itemsets.values())[i][j]
        
            temp[j].append(val)
    
    return temp


# In[16]:


def table3_gen(t):
    T1 = '{'+t[0][0]+'}'
    T2 = T1 + '\n{'+t[0][1]+'}'
    T3 = T2 + '\n{'+t[0][2]+'}'
    T4 = T3 + '\n{'+t[0][3]+'}'
    T5 = T2 + '\n{'+t[0][4]+'}'
    T6 = T5 + '\n{'+t[0][5]+'}'
    return [[T1,T2,T3,T4,T5,T6]]


# In[17]:


'''
#!/usr/bin/env python
import PySimpleGUI as sg
menu_def = [['Options',['Open File', 'Exit']]]


#layout = [[sg.Menu(menu_def)],
#            [sg.Text('Choose a file from the options menu', font='Any 18')],
#         ]

layout = [[sg.Text('Enter 2 files and BPM')],
          [sg.Text('Condition File',  size=(15, 1)), sg.Input(), sg.FileBrowse()],
          [sg.Text('Excel File',  size=(15, 1)), sg.Input(), sg.FileBrowse()],
          [sg.Text('Process Model IMG',size=(15,1)),sg.Input(),sg.FileBrowse()],  
          [sg.Submit()]]    

window = sg.Window('Table', layout,  grab_anywhere=False)


flag = True

while True:
    event, values = window.read()
    # --- Process buttons --- #
    print(event, values)    
    if event in (sg.WIN_CLOSED, 'Exit'):
        break 
            
    elif event == 'Open File' or 'Submit':
        
        if event == "Open File":
            filename = None
            layout = [[sg.Text('Enter 2 files and BPM')],
          [sg.Text('Condition File', size=(15, 1)), sg.Input(), sg.FileBrowse()],
          [sg.Text('Excel File',  size=(15, 1)), sg.Input(), sg.FileBrowse()],
          [sg.Text('Process Model IMG',size=(15,1)),sg.Input(),sg.FileBrowse()],  
          [sg.Submit()]] 
            window1 = sg.Window('Table', layout,  grab_anywhere=False)
            window.close()
            window=window1
            
        else:
            file_event = values[0]
            filename = values[1]
        #filename = sg.popup_get_file( 'Enter the path to the excel file', file_types=(("Data Files", "*.xlsx"),))
        #file_event = sg.popup_get_file( 'Enter the path to the event excel file', file_types=(("Data Files", "*.xlsx"),))
        if filename is not None:
            #  try:
            df2 = file_open(filename,file_event)
                    
            itemSetList,freqItemSet,rules= fetch_itemsets(df2)

            itemsets = itemsdict(df2)

            rules = CMrules(rules,itemSetList) 
            
            t2 = table2_obj(itemsets)
            
            t3 = table3_gen(t2)
            
            layout = [[sg.Menu(menu_def)],
                        [sg.Text('Headers corresponds to the Event Logs followed by their Post conditons ', font='Any 18')],
                        [sg.Table(key='-t1-', values=t1_tweak(itemsets),
                            headings=list(itemsets),
                            auto_size_columns=True,
                          #  vertical_scroll_only = False,

                        )],
                        [sg.Text('Table 2', font='Any 18')],
                        [sg.Table(key='-t2-', values=t2,
                            headings=list(itemsets),
                            auto_size_columns=True,
                            #vertical_scroll_only = False,
                            font = 'any 11'
                        )],
                        [sg.Text('Table 3', font='Any 18')],
                        [sg.Table(key='-t3-', values=t3,
                            headings=list(itemsets),
                            auto_size_columns=True,
                           # vertical_scroll_only = False,
                            col_widths = 60,
                            font = 'any 11'
                        )],
                          [sg.Image(key='-IMG-')]
                        ]
            window1 = sg.Window('Table', layout,  grab_anywhere=False, finalize=True, keep_on_top=True)
            #window1.finalize()
            window1['-IMG-'].update(data=convert_to_bytes(values[2],resize=))
            window.close()
            window=window1
            
            
                      
window.close()
'''


# In[18]:


#!/usr/bin/env python
import PySimpleGUI as sg
menu_def = [['Options',['Open File', 'Exit']]]


#layout = [[sg.Menu(menu_def)],
#            [sg.Text('Choose a file from the options menu', font='Any 18')],
#         ]

layout = [[sg.Text('Enter 2 files and BPM')],
          [sg.Text('Conditions File',  size=(15, 1)), sg.Input(), sg.FileBrowse()],
          [sg.Text('Events File',  size=(15, 1)), sg.Input(), sg.FileBrowse()],
          [sg.Text('Process Model IMG',size=(15,1)),sg.Input(),sg.FileBrowse()],  
          [sg.Submit()]]    

window = sg.Window('Table', layout,  grab_anywhere=False)


flag = True

while True:
    event, values = window.read()
    # --- Process buttons --- #
    print(event, values)    
    if event in (sg.WIN_CLOSED, 'Exit'):
        break 
            
    elif event == 'Open File' or 'Submit':
        
        if event == "Open File":
            filename = None
            layout = [[sg.Text('Enter 2 files and BPM')],
          [sg.Text('Condition File', size=(15, 1)), sg.Input(), sg.FileBrowse()],
          [sg.Text('Excel File',  size=(15, 1)), sg.Input(), sg.FileBrowse()],
          [sg.Text('Process Model IMG',size=(15,1)),sg.Input(),sg.FileBrowse()],  
          [sg.Submit()]] 
            window1 = sg.Window('Table', layout,  grab_anywhere=False)
            window.close()
            window=window1
            
        else:
            file_event = values[0]
            filename = values[1]
        #filename = sg.popup_get_file( 'Enter the path to the excel file', file_types=(("Data Files", "*.xlsx"),))
        #file_event = sg.popup_get_file( 'Enter the path to the event excel file', file_types=(("Data Files", "*.xlsx"),))
        if filename is not None:
            #  try:
            df2 = file_open(filename,file_event)
                    
            itemSetList,freqItemSet,rules= fetch_itemsets(df2)

            itemsets = itemsdict(df2)

            rules = CMrules(rules,itemSetList) 
            
            t2 = table2_obj(itemsets)
            
            t3 = table3_gen(t2)
            tab1_layout =  [[sg.Text('Headers corresponds to the Event Logs followed by their Post conditons ', font='Any 18')],
                        [sg.Table(key='-t1-', values=t1_tweak(itemsets),
                            headings=list(itemsets),
                            auto_size_columns=True,
                          #  vertical_scroll_only = False,

                        )],
                        [sg.Text('Table 2', font='Any 18')],
                        [sg.Table(key='-t2-', values=t2,
                            headings=list(itemsets),
                            auto_size_columns=True,
                            #vertical_scroll_only = False,
                        )],
                        [sg.Text('Table 3', font='Any 18')],
                        [sg.Table(key='-t3-', values=t3,
                            headings=list(itemsets),
                            auto_size_columns=True,
                           # vertical_scroll_only = False,
                            col_widths = 60,
                        )]]    

            tab2_layout = [[sg.Image(key='-IMG-')]]
            
            layout = [[sg.Menu(menu_def)],[sg.TabGroup([[sg.Tab('Tables', tab1_layout), 
                                                         sg.Tab('Image', tab2_layout)]])]                          
                        ]
            window1 = sg.Window('Table', layout,  grab_anywhere=False, finalize=True, keep_on_top=True)
            #window1.finalize()
            window1['-IMG-'].update(data=convert_to_bytes(values[2],resize=(600,600)))
            window.close()
            window=window1
            
            
                      
window.close()


# In[ ]:





# In[ ]:




