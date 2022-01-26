###############################################################################
#
# Description : This program is used extract data from Salesforce
# for segmentation analysis
# Author : Ali Naama
# https://vpsn-99.medium.com/creating-csv-extract-from-salesforce-using-python-quick-and-dirty-learning-4797ab1bead8
# Date : 24/01/2022
#
###############################################################################

import pandas as pd
# logging mngt
import logging
import datetime
from time import sleep
import re
import pandas as pd

# Salesforce mngt
import json
import numpy as np
import re

from simple_salesforce import Salesforce,SalesforceLogin,SFType
from salesforce_bulk import SalesforceBulk
from salesforce_bulk.util import IteratorBytesIO
from urllib.parse import urlparse
from salesforce_bulk import CsvDictsAdapter
import numpy as np


# 0.1 Log mngt

now = datetime.datetime.now()
logging.basicConfig(filename='\log\ExtractSF.log', level=logging.INFO)
logging.info('Started : ' + now.strftime("%Y-%m-%d %H:%M:%S"))

query = 'select Name,SBQQ__ProductName__c,SBQQ__Quantity__c,	Description__c,	Code_Produit__c,	SBQQ__Account__c,	SBQQ__Contract__c,	SBQQ__ContractNumber__c,	' \
        'SBQQ__StartDate__c,	SBQQ__EndDate__c,	SBQQ__NetPrice__c,	SBQQ__Contract__r.BillingCountry from SBQQ__Subscription__c  where  ' \
        'CreatedDate >= 2020-01-01T00:00:00Z and CreatedDate < 2021-01-01T00:00:00Z order by SBQQ__Contract__c, CreatedDate desc'

# Data Migration - User - Quality
sf = Salesforce(password='', username='', organizationId='', domain='test')
data = sf.bulk.SBQQ__Subscription__c.query_all(query)
#dataframe = pd.dataframe(data['records'])

df = pd.DataFrame.from_dict(data,orient='columns').drop('attributes',axis=1)
writer = pd.ExcelWriter('\data\ExtractSales2020.xlsx', engine='xlsxwriter')
df.to_excel(writer, sheet_name='Sales', index=False)
# Close the Pandas Excel writer and output the Excel file.
writer.save()
