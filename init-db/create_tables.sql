-- init-db/create_tables.sql

-- 1. Create Tables
CREATE TABLE IF NOT EXISTS disposition (
    disp_id VARCHAR(50) PRIMARY KEY,
    client_id VARCHAR(50),
    account_id VARCHAR(20),
    type VARCHAR(50) 
);

CREATE TABLE IF NOT EXISTS loan (
    loan_id VARCHAR(20) PRIMARY KEY,
    account_id VARCHAR(20),
    amount NUMERIC,
    duration INT,
    payments NUMERIC,
    status VARCHAR(5),
    year INT,
    month INT,
    day INT,
    fulldate DATE,
    location VARCHAR(50),
    purpose VARCHAR(50)
);

CREATE TABLE IF NOT EXISTS trans (
    id SERIAL PRIMARY KEY,
    trans_id VARCHAR(20),
    account_id VARCHAR(20),
    type VARCHAR(20),
    operation VARCHAR(50),
    amount NUMERIC,
    balance NUMERIC,
    k_symbol VARCHAR(50),
    bank VARCHAR(50),
    account VARCHAR(50),
    year INT,
    month INT,
    day INT,
    fulldate DATE,
    fulltime TIME,
    fulldatewithtime TIMESTAMP
);

CREATE TABLE IF NOT EXISTS crmcallcenterlogs (
    date_received DATE,
    complaint_id VARCHAR(20),
    rand_client VARCHAR(20),
    phonefinal VARCHAR(20),
    vru_line VARCHAR(20),
    call_id INT,
    priority INT,
    type VARCHAR(50),
    outcome VARCHAR(50),
    server VARCHAR(50),
    ser_start TIME,
    ser_exit TIME,
    ser_time INTERVAL
);

CREATE TABLE IF NOT EXISTS crmevents (
    date_received DATE,
    product VARCHAR(100),
    sub_product VARCHAR(100),
    issue VARCHAR(200),
    sub_issue VARCHAR(200),
    consumer_complaint_narrative TEXT,
    tags VARCHAR(200),
    consumer_consent_provided VARCHAR(50),
    submitted_via VARCHAR(50),
    date_sent_to_company DATE,
    company_response_to_consumer VARCHAR(200),
    timely_response VARCHAR(10),
    consumer_disputed VARCHAR(10),
    complaint_id VARCHAR(20),
    client_id VARCHAR(20)
);

CREATE TABLE IF NOT EXISTS crmreviews (
    date DATE,
    stars INT,
    reviews TEXT,
    product VARCHAR(200),
    district_id INT
);



-- 2. Load Data
\copy disposition FROM '/docker-entrypoint-initdb.d/completeddisposition.csv' DELIMITER ',' CSV HEADER;
\copy loan FROM '/docker-entrypoint-initdb.d/completedloan.csv' DELIMITER ',' CSV HEADER;
\copy crmcallcenterlogs FROM '/docker-entrypoint-initdb.d/CRMCallCenterLogs.csv' DELIMITER ',' CSV HEADER;
\copy crmevents FROM '/docker-entrypoint-initdb.d/CRMEvents.csv' DELIMITER ',' CSV HEADER;
\copy crmreviews FROM '/docker-entrypoint-initdb.d/CRMReviews.csv' DELIMITER ',' CSV HEADER;
\copy trans(id, trans_id, account_id, type, operation, amount, balance, k_symbol, bank, account, year, month, day, fulldate, fulltime, fulldatewithtime)
FROM '/docker-entrypoint-initdb.d/completedtrans.csv' DELIMITER ',' CSV HEADER;