-- Creating AWS account 
-- Goto the website >> https://aws.amazon.com/console/

-- Sign-in as Root user

Fraud Detection model data ELT pipeline using the AWS S3 Bucket.

--===============================================
Navigation: Amazon S3 > Buckets > Create bucket

S3 bucket name: fraud-data-csv
folder name: stg_csv_file
file name: Fraud.csv

--In S3 bucket file location
S3 URL: s3://fraud-data-csv/stg_csv_file/

--Now creating user to access the file in S3 bucket

IAM-->user-->Permission
so we are creating user for the snowflakes and the user will have the access over the bucket so providing permission (AmazonS3FullAccess)

Now creating IAM user
user name : FraudDataRead
Navigation: IAM > Users > FraudDataRead
ARN: arn:aws:iam::086877291111:user/FraudDataRead
Amazon Resource Number

IAM --> Users --> FraudDataRead --> Create access key
So access key is required by snowflakes to access the file in the s3 bucket 

Access key
If you lose or forget your secret access key, you cannot retrieve it. Instead, create a new access key and make the old key inactive.

Access key : AKIARIOSKIDDIWIAAAA
Secret access key : FO1Ep7AdlUa3u89EqwRrQSKhqUxrJAAAAAAAAA

Create the Role:
IAM > Roles
RoleName: FraudDataRead
ARN: arn:aws:iam::08687729AAAA:role/FraudDataRead

--Updated the trust relationship after creating the storage integration
--Sample trust raltationship
-- {
--     "Version": "2012-10-17",
--     "Statement": [
--         {
--             "Effect": "Allow",
--             "Principal": {
--                 "AWS": "Snowflakes storage integration: STORAGE_AWS_IAM_USER_ARN"
--             },
--             "Action": "sts:AssumeRole",
--             "Condition": {
--                 "StringEquals": {
--                     "sts:ExternalId": "Snowflakes storage integration: STORAGE_AWS_EXTERNAL_ID"
--                 }
--             }
--         }
--     ]
-- }

--============================================================
-- After coming to snowflakes creating stage, file format
show stages;

show integrations;

show file formats;

-- storage_aws_role_arn = iam_role_arn
-- iam  --> Identity access management 
-- role --> FraudDataRead
-- arn  --> arn:aws:iam::086877291111:role/FraudDataRead

create storage integration s3_integration
    type = external_stage
    storage_provider = s3
    storage_aws_role_arn = 'arn:aws:iam::086877291111:role/FraudDataRead'
    enabled = true
    storage_allowed_locations = ( 's3://fraud-data-csv/stg_csv_file/' )
     comment = 'This integration will help you to access file in aws'
    ;
    

desc storage integration s3_integration;
-- STORAGE_ALLOWED_LOCATIONS		s3://fraud-data-csv/stg_csv_file/
-- STORAGE_AWS_IAM_USER_ARN		arn:aws:iam::552829149111:user/a09c1000-s
-- STORAGE_AWS_ROLE_ARN	    		arn:aws:iam::086877290111:role/FraudDataRead
-- STORAGE_AWS_EXTERNAL_ID	 	FS97210_SFCRole=2_1ofGhcRjunXeAAAAAAClbw++I9o=

--Before creating stage, need to update the trust relationships in AWS for the user FraudDataRead
CREATE STAGE STG_S3_FILES 
	URL = 's3://fraud-data-csv/stg_csv_file/' 
	STORAGE_INTEGRATION = s3_integration    
	DIRECTORY = ( ENABLE = true ) 
	COMMENT = 'This stage will be used to access csv files from aws';

    
desc STAGE STG_S3_FILES;

create or replace file format csv_format
type =csv
skip_header=1
field_delimiter=',' ;  

list @STG_S3_FILES; 

--column in csv file: step, type, amount, nameOrig, oldbalanceOrg, newbalanceOrig, nameDest, oldbalanceDest, newbalanceDest, isFraud, isFlaggedFraud;

--======================================================================
-- Creating table : to automatically load the data updation while file upload 

create or replace TABLE Fraud_transcation_History (
step NUMBER,
Trans_type varchar,
amount number,
nameOrig varchar,
oldbalanceOrg number,
newbalanceOrig number,
nameDest varchar,
oldbalanceDest number,
newbalanceDest number,
isFraud	varchar,
isFlaggedFraud varchar );

--Checking the data from stage location
select top 10 $1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11 from @STG_S3_FILES;

--To load the data into the table
copy into Fraud_transcation_History from @STG_S3_FILES file_format=csv_format;

----================================================================

--Creating pipe: That automatically load the data from file upload in the S3 bucket
show pipes;

-- auto_ingest = true : automatically load the data
create pipe pipe_load_s3data
auto_ingest = true 
as
copy into Fraud_transcation_History from @STG_S3_FILES file_format=csv_format;

--=============================================================

--Verify the data :
select top 10 * from Fraud_transcation_History;

--Verifing using the dbt (creating model)
select * from DEV_DB.DBT_SCHEMA.Fraud_transcation_Record;



