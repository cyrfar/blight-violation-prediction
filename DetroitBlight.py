def blight_model():
    import numpy as np
    import pandas as pd
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.preprocessing import StandardScaler
    
    #merge training set with addresses and coordinates
    train = pd.read_csv('train.csv', encoding='mac_roman')
    test = pd.read_csv('test.csv')
    addresses = pd.read_csv('addresses.csv')
    latlons = pd.read_csv('latlons.csv')
    merged = pd.merge(addresses, latlons, on ='address', how ='inner')
    train = train.merge( merged, on ='ticket_id', how = 'inner');
    test = test.merge(merged, on ='ticket_id', how = 'inner');

    #contruct a list of typos for the word Detroit appearing in the 
    #train['city'] column. 
    Detroit_with_typos = set(
    ['DETROIT', 'Detroit', 'DETORIT', 'DETROT', 'Detrot', 'detroit', 'detrot',
     'DET', 'DETROI', 'DETROIR', 'DET.', 'DETRIOIT', 'Det', 'D', 'det', 'd',
     'DERTROIT', 'DETRIOT', 'DETRIT', 'Deroit', 'detriot', 'DEROT', 'DETRPOT',
     'Detorit','DETRTOI', 'DET,', 'DETR4OIT', 'DETRIUT', 'DEROIT', 'DEt',
     'Det.', 'DETOIT', 'DERTOIT',  'DEETROIT', 'DETROTI', 'DTEROIT','DETROIOT',
     'DETROIT1','DETROITQ', 'DET ,', 'DET ROIT', 'DET,.', 'DET. MI.', 'DET.,',
     'DET., MI.', 'DETAROIT', 'DETEOIT', 'DETEROIT', 'DETORIIT', 'DETREOIT',
     'DTROIT', 'DETRIOIT', 'DETRJOIT', 'DETROIIT', 'DETROIRT', 'DETROIS',
     'DETROIT  4', 'DETROIT,', 'DETROIT, MI.', 'DETROIT, MI. 48206', 'DETROITF',
     'DETROITI', 'DETROITL', 'DETROITM', 'DETROITT', 'DETROITY','DETROIT`', 
     'DETROITdetroit', 'DETROIYT', 'DETROKT', 'DETROOIT', 'DETROPIT',
     'DETROUIT', 'DETRROIT', 'DETTROIT', 'DETTROITM', 'DETYROIT', 'DETZVILLE',
     'Det', 'Detrioit', 'Detriot', 'Detriut','Detro;it', 'Detrofit', 'Detroi', 
     'Detroir', 'Detroit`', 'Detroitf', 'Detroitli', 'Detroti', 'Detrroit',
     'Dteroit', 'dteroit', 'DEtroit', 'cetroit', 'deroit', 'dertoit', 
     'deteroit', 'detoit', 'detrtoit', 'BELOIT', 'CETROIT', 'DEEEEETROIT',
     'DERROIT', 'DEYROIT', 'DFETROIT', 'DKETROIT', 'DRTROIT', 'DWETROIT',
     'EAST  DETROIT', 'ETROIT','WARRENDETROIT', 'dETROIT','det.', 'det48234',
     'detroiit', 'detroiot', 'detroir', 'detroit,mi', 'detroit`', 'detroitt',
     'detrorit', 'DE',  'DD', 'DDD', 'DDDDDD', 'DDDDDDD', 'DDDDDDDD', 'deT'])

    #These columns will be dropped due to potential leakage
    data_leakage = set(
    ['payment_amount', 'payment_date', 'payment_status', 'balance_due',
     'collection_status', 'compliance_detail'])
     
    #These columns will also be dropped
    drop_these = [
    'inspector_name', 'violation_street_number', 'violation_street_name',
    'violation_zip_code', 'violation_description', 'violator_name', 
    'violation_code', 'ticket_issued_date', 'hearing_date', 
    'mailing_address_str_number', 'mailing_address_str_name', 'city', 'state',
    'judgment_amount','zip_code', 'non_us_str_code', 'country', 
    'grafitti_status', 'admin_fee', 'state_fee', 'late_fee', 'discount_amount',
    'clean_up_cost', 'address', 'agency_name']

    #Relevant variables split into categorical, numeric
    cat_vars = ['disposition', 'mailing_non_USA', 'mailing_non_DT',
                'mailing_non_MI']
    num_vars = ['fine_amount', 'days_from_issue_to_hearing', 'lat', 'lon']
    
    
    #Elements in set(X_test['disposition']) that will be replaced
    replace = ['Responsible - Compl/Adj by Determi', 'Responsible by Dismissal', 
               'Responsible (Fine Waived) by Admis', 
               'Responsible - Compl/Adj by Default']
    
    
    def preprocess(train, test):  
        '''Create new fields and clean the train and test data'''
        for df in [train, test]: 
            #create new features for the mailing address
            df['mailing_non_USA'] = \
            [1 if country not in ['USA'] else 0 for country in df['country']]
            df['mailing_non_DT'] = \
            [1 if city not in Detroit_with_typos else 0 for city in df['city']]
            df['mailing_non_MI'] = \
            [1 if state not in ['MI'] else 0 for state in df['state']]

            #create a new feature: the number of days elaspsed from
            #the ticket-issue-day to the hearing day
            df['ticket_issued_date'] = pd.to_datetime(df['ticket_issued_date'])
            df['hearing_date'] = pd.to_datetime(df['hearing_date'])
            delta = df['hearing_date'] - df['ticket_issued_date']
            df.loc[delta.notnull(), 'days_from_issue_to_hearing'] = delta.dt.days
                
            #drop potential data leaking fields
            if set(df.columns).intersection(data_leakage):
                df = df.drop(data_leakage, axis = 1)
                
            #drop these irrelevant fields, or fields with mostly NA's in them
            df = df.drop(drop_these, axis = 1)
        
            #cleaning missing data 
            mean = np.mean(df['days_from_issue_to_hearing'])
            df['days_from_issue_to_hearing'].fillna(mean, inplace = True)
            #df['violation_code'].fillna(' ', inplace = True)
            df['fine_amount'].fillna(np.mean(df['fine_amount']), inplace = True)
            df['lat'].fillna(np.mean(df['lat']), inplace = True)
            df['lon'].fillna(np.mean(df['lon']), inplace = True)

            if set(df.columns).intersection(set(['compliance'])):
                #trim traininig data:
                df = df.dropna(axis=0)
                df.reset_index(inplace=True, drop=True)
                X_train = df.drop('compliance',axis =1)            
                y_train = pd.DataFrame(df['compliance'])
            else: 
                #trim test data:
                #Replace elements of set(X_test['disposition']) that don't appear
                #in set(X_train['disposition']), with 'Responsible by Default'.
                df['disposition'].replace(replace, 'Responsible by Default', 
                                          inplace=True)
                X_test = df
        return X_train, X_test, y_train


    def EncoderScaler(X_train, X_test):
        X1 = pd.get_dummies(X_train[cat_vars])
        scaler = StandardScaler().fit(X_train[num_vars])
        X2 = pd.DataFrame(scaler.transform(X_train[num_vars]), columns = num_vars)
        X_train_map = X1.join(X2)
        mean =np.mean(X_train_map['days_from_issue_to_hearing'])
        X_train_map['days_from_issue_to_hearing'].fillna(mean, inplace = True)
        X_train_map['lat'].fillna(np.mean(X_train_map['lat']), inplace = True)
        X_train_map['lon'].fillna(np.mean(X_train_map['lon']), inplace = True)
        X_train_map['fine_amount'].fillna(np.mean(X_train_map['fine_amount']),
         inplace = True)
        X_train_map.reset_index(inplace=True, drop=True)
        X3 = pd.get_dummies(X_test[cat_vars])
        X4 = pd.DataFrame(scaler.transform(X_test[num_vars]), columns = num_vars)
        X_test_map = X3.join(X4)
        return X_train_map, X_test_map

    #get data with the relevant features and process it
    X_train, X_test, y_train = preprocess(train, test)
    X_train_map, X_test_map = EncoderScaler(X_train, X_test)
    
    #calculate probabilities using a gradient boost classifier
    grd = GradientBoostingClassifier(n_estimators=500, min_samples_split=50,
                     max_depth = 3, min_samples_leaf=10, learning_rate=0.05)
    grd.fit(np.array(X_train_map),np.array(y_train))
    y_pred = grd.predict_proba(np.array(X_test_map))
    
    return  pd.Series(y_pred[:,1], index =X_test['ticket_id'].tolist(),
     name='compliance')
