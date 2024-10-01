# Write a funciton to preprocess the data

def preprocess_data(df):
    
    # Remove the rows with missing values
    df = df.dropna()
    
    # Convert the 'Possibility' column to binary values
    df['Possibility'] = df['Possibility'].map({'Yes': 1, 'No': 0})
    
    
    # Create capitalloss_binary and capitalgain_binary columns by assigning 1 if the value is greater than 0, otherwise 0
    df['capitalloss_binary'] = df['capitalloss'].apply(lambda x: 1 if x > 0 else 0)
    df['capitalgain_binary'] = df['capitalgain'].apply(lambda x: 1 if x > 0 else 0)
    
    # Perform label encoding on the 'education' column
    df['education_labelencoding'] = df['education'].map({
        'Preschool': 0, 
        '1st-4th ': 1,
        '5th-6th': 1,
        '7th-8th ': 1,
        '9th':1,
        '10th': 2,
        '11th': 2,
        '12th': 2,
        'HS-grad': 3,
        'Some-college': 4,
        'Assoc-acdm': 5,
        'Assoc-voc': 5,
        'Bachelors': 6,
        'Masters': 7,
        'Doctorate': 8,
        'Prof-school': 8})
    

    
    # Perform binary encoding on the 'maritalstatus' column
    df['maritalstatus_binary'] = df['maritalstatus'].map({
        'Never-married': 0,
        'Divorced': 0,
        'Separated': 0,
        'Widowed': 0,
        'Married-spouse-absent': 0,
        'Married-civ-spouse': 1,
        'Married-AF-spouse': 1})
    
    # Define custom categories based on the analysis
    df['maritalstatus_group'] = df['maritalstatus'].map({
        'Married-AF-spouse': 'Married',
        'Married-civ-spouse': 'Married',
        'Divorced': 'Divorced-Widowed-Abs',
        'Widowed': 'Divorced-Widowed-Abs',
        'Married-spouse-absent': 'Divorced-Widowed-Abs',
        'Separated': 'Separated-Never-Married',
        'Never-married': 'Separated-Never-Married'
    })
    
    # # Perform one-hot encoding on the new 'maritalstatus_group' column
    # df = pd.get_dummies(df, columns=['maritalstatus_group'], drop_first=True)

    # Perform binary encoding on the 'sex' column
    df['sex'] = df['sex'].map({'Male':1, 'Female':0})
    
    # Performing label encoding on the 'occupation' column
    # Define custom groupings and label encoding for 'occupation'
    df['occupation_encoded'] = df['occupation'].map({
        'Exec-managerial': 5,
        'Prof-specialty': 5,
        'Protective-serv': 4,
        'Tech-support': 4,
        'Sales': 4,
        'Craft-repair': 3,
        'Transport-moving': 3,
        'Adm-clerical': 2,
        'Machine-op-inspct': 2,
        'Farming-fishing': 2,
        'Armed-Forces': 2,
        'Handlers-cleaners': 1,
        'Other-service': 1,
        'Priv-house-serv': 0
    })
    
    # Step 1: Combine 'Amer-Indian-Eskimo' and 'Other' into 'other_new'
    df['race_modified'] = df['race'].replace({
        'Amer-Indian-Eskimo': 'other_new',
        'Other': 'other_new'
    })
    
    df['native_labelencoding'] = df['native'].map({
        # 0-5% range
        'Dominican-Republic': 0,
        'Outlying-US(Guam-USVI-etc)': 0,
        'Columbia': 0,
        'Guatemala': 0,
        
        # 5-10% range
        'Mexico': 1,

        'Nicaragua': 1,
        'Peru': 1,
        'Vietnam': 1,
        'Honduras': 1,
        'El-Salvador': 1,
        'Haiti': 1,

        
        # 10-15% range
        'Puerto-Rico': 2,
        'Trinadad&Tobago': 2,
        'Portugal': 2,
        'Laos': 2,
        'Ecuador': 2,
        'Jamaica': 2,

        
        # 15-20% range
        'Thailand': 3,
        'Ireland': 3,
        'South': 3,
        'Scotland': 3,
        'Poland': 3,
        
        # 20-25% range
        'Hungary': 4,
        'United-States': 4,
        
        # 25-30% range
        'Cuba': 5,
        'China': 5,
        'Greece': 5,

        
        # 30-35% range
        'Philippines': 6,
        'Hong': 6,
        'Canada': 6,
        'Germany': 6,
        'England': 6,
        'Italy': 6,
        
        # 35-40% range
        'Yugoslavia': 7,
        'Cambodia': 7,
        'Japan': 7,
        
        # 40-45% range
        'India': 8,
        'Iran': 8,
        'France': 8,
        'Taiwan': 8
    })
    
    

# Remove the rows with missing values
df = df.dropna()

# Convert the 'Possibility' column to binary values
df['Possibility'] = df['Possibility'].map({'Yes': 1, 'No': 0})


# Create capitalloss_binary and capitalgain_binary columns by assigning 1 if the value is greater than 0, otherwise 0
# df['capitalloss_binary'] = df['capitalloss'].apply(lambda x: 1 if x > 0 else 0)
df['capitalgain_binary'] = df['capitalgain'].apply(lambda x: 1 if x > 0 else 0)

# Perform label encoding on the 'education' column
df['education_labelencoding'] = df['education'].map({
    'Preschool': 0, 
    '1st-4th': 1,
    '5th-6th': 1,
    '7th-8th': 1,
    '9th':1,
    '10th': 2,
    '11th': 2,
    '12th': 2,
    'HS-grad': 3,
    'Some-college': 4,
    'Assoc-acdm': 5,
    'Assoc-voc': 5,
    'Bachelors': 6,
    'Masters': 7,
    'Doctorate': 8,
    'Prof-school': 8})



# Perform binary encoding on the 'maritalstatus' column
df['maritalstatus_binary'] = df['maritalstatus'].map({
    'Never-married': 0,
    'Divorced': 0,
    'Separated': 0,
    'Widowed': 0,
    'Married-spouse-absent': 0,
    'Married-civ-spouse': 1,
    'Married-AF-spouse': 1})

# Define custom categories based on the analysis
df['maritalstatus_group'] = df['maritalstatus'].map({
    'Married-AF-spouse': 'Married',
    'Married-civ-spouse': 'Married',
    'Divorced': 'Divorced-Widowed-Abs',
    'Widowed': 'Divorced-Widowed-Abs',
    'Married-spouse-absent': 'Divorced-Widowed-Abs',
    'Separated': 'Separated-Never-Married',
    'Never-married': 'Separated-Never-Married'
})

# # Perform one-hot encoding on the new 'maritalstatus_group' column
# df = pd.get_dummies(df, columns=['maritalstatus_group'], drop_first=True)

# Perform binary encoding on the 'sex' column
df['sex'] = df['sex'].map({'Male':1, 'Female':0})

# Performing label encoding on the 'occupation' column
# Define custom groupings and label encoding for 'occupation'
df['occupation_encoded'] = df['occupation'].map({
    'Exec-managerial': 5,
    'Prof-specialty': 5,
    'Protective-serv': 4,
    'Tech-support': 4,
    'Sales': 4,
    'Craft-repair': 3,
    'Transport-moving': 3,
    'Adm-clerical': 2,
    'Machine-op-inspct': 2,
    'Farming-fishing': 2,
    'Armed-Forces': 2,
    'Handlers-cleaners': 1,
    'Other-service': 1,
    'Priv-house-serv': 0
})

# Step 1: Combine 'Amer-Indian-Eskimo' and 'Other' into 'other_new'
df['race_modified'] = df['race'].replace({
    'Amer-Indian-Eskimo': 'other_new',
    'Other': 'other_new'
})

df['native_labelencoding'] = df['native'].map({
    # 0-5% range
    'Dominican-Republic': 0,
    'Outlying-US(Guam-USVI-etc)': 0,
    'Columbia': 0,
    'Guatemala': 0,
    
    # 5-10% range
    'Mexico': 1,

    'Nicaragua': 1,
    'Peru': 1,
    'Vietnam': 1,
    'Honduras': 1,
    'El-Salvador': 1,
    'Haiti': 1,

    
    # 10-15% range
    'Puerto-Rico': 2,
    'Trinadad&Tobago': 2,
    'Portugal': 2,
    'Laos': 2,
    'Ecuador': 2,
    'Jamaica': 2,

    
    # 15-20% range
    'Thailand': 3,
    'Ireland': 3,
    'South': 3,
    'Scotland': 3,
    'Poland': 3,
    
    # 20-25% range
    'Hungary': 4,
    'United-States': 4,
    
    # 25-30% range
    'Cuba': 5,
    'China': 5,
    'Greece': 5,

    
    # 30-35% range
    'Philippines': 6,
    'Hong': 6,
    'Canada': 6,
    'Germany': 6,
    'England': 6,
    'Italy': 6,
    
    # 35-40% range
    'Yugoslavia': 7,
    'Cambodia': 7,
    'Japan': 7,
    
    # 40-45% range
    'India': 8,
    'Iran': 8,
    'France': 8,
    'Taiwan': 8
})