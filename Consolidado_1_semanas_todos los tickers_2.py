# -*- coding: utf-8 -*-
"""
Consolidado 5 semanas (una sola solapa)
- Descarga datos una sola vez
- Calcula indicadores "Gemini" + "ChatGPT"
- Backtest rápido de 5 semanas (1 predicción por semana)
- Consolida TODO en una única hoja de Excel
"""

import warnings, os
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path

import yfinance as yf
import pandas_ta as ta
from sklearn.ensemble import RandomForestClassifier

# ================== CONFIG ==================
# Usá tu lista larga si querés. Asegúrate que incluya 'SPY'
TICKERS = [
   
   'SPY','AACB', 'AACBR', 'AACBU', 'AACG', 'AACIU', 'AACOU', 'AAM', 'AAME', 'AAPG', 'AAUC', 'ABI', 'ABLV', 'ABLVW', 'ABOS', 'ABP', 'ABPWW', 'ABTC', 'ABTS', 'ABVC', 'ABVE', 'ABVEW', 'ABVX', 'ABX', 'ABXL', 'ACB', 'ACCL', 'ACCS', 'ACET', 'ACFN', 'ACGLN', 'ACGLO', 'ACH', 'ACHC', 'ACHV', 'ACII', 'ACIU', 'ACOG', 'ACON', 'ACONW', 'ACP', 'ACRV', 'ACV', 'ACWX', 'ACXP', 'AD', 'ADAC', 'ADACU', 'ADACW', 'ADAG', 'ADAMG', 'ADAMH', 'ADAMI', 'ADAML', 'ADAMM', 'ADAMN', 'ADAMO', 'ADAMZ', 'ADGM', 'ADIL', 'ADSE', 'ADSEW', 'ADT', 'ADTX', 'ADUR', 'ADVB', 'ADX', 'ADXN', 'AEAQ', 'AEAQU', 'AEAQW', 'AEC', 'AEF', 'AEFC', 'AEG', 'AEHL', 'AEI', 'AEMD', 'AENT', 'AENTW', 'AEON', 'AERO', 'AERT', 'AERTW', 'AEVAW', 'AEXA', 'AFB', 'AFBI', 'AFG', 'AFGB', 'AFGC', 'AFGD', 'AFGE', 'AFJK', 'AFJKR', 'AFJKU', 'AFRIW', 'AFYA', 'AGAE', 'AGBK', 'AGCC', 'AGD', 'AGEN', 'AGH', 'AGI', 'AGIG', 'AGM.A', 'AGMB', 'AGMH', 'AGNC', 'AGNCL', 'AGNCM', 'AGNCN', 'AGNCO', 'AGNCP', 'AGNCZ', 'AGO', 'AGPU', 'AGRO', 'AGRZ', 'AHG', 'AHMA', 'AHT', 'AIA', 'AIFF', 'AIFU', 'AIHS', 'AIIA', 'AIIO', 'AIIOW', 'AIM', 'AIMD', 'AIMDW', 'AIO', 'AIOS', 'AIRE', 'AIRG', 'AIRI', 'AIRJW', 'AIRT', 'AIRTP', 'AIS', 'AISPW', 'AIT', 'AIXC', 'AIXI', 'AIZN', 'AKA', 'AKAM', 'AKAN', 'AKO.A', 'AKO.B', 'AKTS', 'AKTX', 'AL', 'ALAB', 'ALAR', 'ALBT', 'ALC', 'ALCY', 'ALCYW', 'ALDF', 'ALDFU', 'ALDFW', 'ALF', 'ALFUU', 'ALFUW', 'ALGS', 'ALH', 'ALIS', 'ALISR', 'ALISU', 'ALLR', 'ALLT', 'ALM', 'ALOT', 'ALOVU', 'ALPS', 'ALTO', 'ALTS', 'ALUB', 'ALUR', 'ALVO', 'ALVOW', 'ALXO', 'ALZN', 'AM', 'AMBO', 'AMBR', 'AMCI', 'AMID', 'AMIX', 'AMOD', 'AMODW', 'AMPG', 'AMPGR', 'AMPGZ', 'AMPY', 'AMRN', 'AMRZ', 'AMS', 'AMST', 'AMTD', 'AMTM', 'AMTX', 'AMWL', 'AMZE', 'AN', 'ANDG', 'ANEB', 'ANGH', 'ANGHW', 'ANGX', 'ANIX', 'ANL', 'ANNA', 'ANNAW', 'ANPA', 'ANRO', 'ANSC', 'ANSCU', 'ANSCW', 'ANTA', 'ANTX', 'ANV', 'ANVS', 'ANY', 'AOD', 'AOMD', 'AOMN', 'AP', 'APAC', 'APACR', 'APACU', 'APAD', 'APADR', 'APADU', 'APC', 'APG', 'API', 'APLM', 'APLMW', 'APLS', 'APM', 'APOS', 'APPX', 'APRE', 'APT', 'APUS', 'APVO', 'APWC', 'APXT', 'APXTU', 'APXTW', 'APYX', 'AQB', 'AQMS', 'AQN', 'AQNB', 'ARAI', 'ARB', 'ARBB', 'ARBE', 'ARBEW', 'ARBK', 'ARCIU', 'ARCO', 'ARCX', 'ARDC', 'AREB', 'AREBW', 'AREC', 'ARIS', 'ARKR', 'ARLP', 'ARM', 'ARMH', 'ARMN', 'ARMP', 'ARP', 'ARQQ', 'ARQQW', 'ARTCU', 'ARTL', 'ARTNA', 'ARTV', 'ARTW', 'ARW', 'ARX', 'AS', 'ASA', 'ASBA', 'ASBP', 'ASBPW', 'ASG', 'ASGI', 'ASH', 'ASIA', 'ASMB', 'ASNS', 'ASPC', 'ASPCR', 'ASPCU', 'ASPS', 'ASPSW', 'ASPSZ', 'ASRT', 'ASRV', 'ASST', 'ASTC', 'ASTI', 'ASTL', 'ASTLW', 'ASTX', 'ATAI', 'ATAT', 'ATCH', 'ATCX', 'ATER', 'ATGL', 'ATHE', 'ATHM', 'ATHR', 'ATHS', 'ATII', 'ATIIU', 'ATIIW', 'ATLCL', 'ATLCP', 'ATLCZ', 'ATLN', 'ATLX', 'ATNM', 'ATO', 'ATON', 'ATOS', 'ATPC', 'ATR', 'ATRA', 'ATS', 'ATXG', 'AUBN', 'AUDC', 'AUGO', 'AUID', 'AUNA', 'AUR', 'AURE', 'AUROW', 'AUST', 'AUTL', 'AUUD', 'AVBC', 'AVBH', 'AVIV', 'AVK', 'AVL', 'AVT', 'AVTX', 'AVX', 'AVY', 'AWAY', 'AWF', 'AWI', 'AWP', 'AWRE', 'AWX', 'AXG', 'AXIA', 'AXIL', 'AXIN', 'AXINR', 'AXINU', 'AXR', 'AXS', 'AYI', 'AYTU', 'AZ', 'AZI', 'AZTR', 'B', 'BACC', 'BACCR', 'BACCU', 'BACQ', 'BACQR', 'BACQU', 'BAER', 'BAERW', 'BAFN', 'BAH', 'BAK', 'BALT', 'BAM', 'BANFP', 'BANL', 'BANX', 'BAOS', 'BAP', 'BATL', 'BAYA', 'BAYAR', 'BB', 'BBCQ', 'BBCQU', 'BBCQW', 'BBDC', 'BBDO', 'BBGI', 'BBLG', 'BBLGW', 'BBLU', 'BBN', 'BBOT', 'BBU', 'BC', 'BCAB', 'BCAR', 'BCARU', 'BCARW', 'BCAT', 'BCDA', 'BCE', 'BCG', 'BCGWW', 'BCH', 'BCIC', 'BCOR', 'BCS', 'BCSF', 'BCSS', 'BCTX', 'BCTXL', 'BCTXW', 'BCTXZ', 'BCV', 'BCX', 'BCYC', 'BDCI', 'BDCIU', 'BDCIW', 'BDJ', 'BDL', 'BDMD', 'BDMDW', 'BDRX', 'BDSX', 'BDTX', 'BEAG', 'BEAGR', 'BEAGU', 'BEAT', 'BEATW', 'BEBE', 'BEEM', 'BENF', 'BENFW', 'BEP', 'BEPC', 'BEPH', 'BEPI', 'BEPJ', 'BESS', 'BETA', 'BETRW', 'BEZ', 'BF.A', 'BFAM', 'BFK', 'BFRG', 'BFRGW', 'BFRI', 'BFRIW', 'BFZ', 'BGB', 'BGH', 'BGI', 'BGIN', 'BGL', 'BGLC', 'BGLWW', 'BGM', 'BGMS', 'BGMSP', 'BGR', 'BGSF', 'BGSI', 'BGT', 'BGX', 'BGY', 'BH.A', 'BHAT', 'BHF', 'BHFAL', 'BHFAM', 'BHFAN', 'BHFAO', 'BHFAP', 'BHK', 'BHM', 'BHST', 'BHV', 'BIAF', 'BIAFW', 'BIII', 'BIO.B', 'BIOX', 'BIP', 'BIPH', 'BIPI', 'BIPJ', 'BIRD', 'BIT', 'BITF', 'BITI', 'BITS', 'BIVI', 'BIVIW', 'BIXI', 'BIXIU', 'BIXIW', 'BIYA', 'BJ', 'BJDX', 'BKHA', 'BKHAR', 'BKHAU', 'BKN', 'BKT', 'BKYI', 'BLBX', 'BLCO', 'BLE', 'BLIN', 'BLIV', 'BLLN', 'BLNE', 'BLOX', 'BLRK', 'BLRKU', 'BLRKW', 'BLRX', 'BLSH', 'BLTE', 'BLUW', 'BLUWU', 'BLUWW', 'BLW', 'BLZR', 'BLZRU', 'BLZRW', 'BME', 'BMEA', 'BMEZ', 'BMGL', 'BMHL', 'BMM', 'BMN', 'BMNR', 'BMR', 'BMRA', 'BN', 'BNAI', 'BNAIW', 'BNBX', 'BNC', 'BNCWW', 'BNDW', 'BNGO', 'BNH', 'BNJ', 'BNKK', 'BNR', 'BNRG', 'BNT', 'BNY', 'BNZI', 'BNZIW', 'BOBS', 'BODI', 'BOE', 'BOF', 'BOKF', 'BOLD', 'BOLT', 'BON', 'BOSC', 'BOTJ', 'BOXL', 'BPAC', 'BPACR', 'BPACU', 'BPI', 'BPOP', 'BPOPM', 'BPRE', 'BPYPM', 'BPYPN', 'BPYPO', 'BPYPP', 'BQ', 'BRAG', 'BRAI', 'BRBI', 'BRC', 'BRCB', 'BRFH', 'BRIA', 'BRID', 'BRK.A', 'BRK.B', 'BRKRP', 'BRLS', 'BRLSW', 'BRLT', 'BRN', 'BRNS', 'BROS', 'BRR', 'BRRWW', 'BRTX', 'BRW', 'BSAA', 'BSAAR', 'BSAAU', 'BSAC', 'BSBK', 'BSJS', 'BSL', 'BSM', 'BSMS', 'BSMU', 'BST', 'BSTZ', 'BSY', 'BTA', 'BTAI', 'BTBD', 'BTBDW', 'BTC', 'BTCS', 'BTCT', 'BTE', 'BTF', 'BTGO', 'BTM', 'BTMWW', 'BTO', 'BTOC', 'BTOG', 'BTQ', 'BTSGU', 'BTT', 'BTTC', 'BTX', 'BTZ', 'BUDA', 'BUFF', 'BUG', 'BUI', 'BULL', 'BULLW', 'BURU', 'BUSEP', 'BUUU', 'BVC', 'BW', 'BWAY', 'BWBBP', 'BWEN', 'BWG', 'BWLP', 'BWMX', 'BWNB', 'BWOW', 'BXMX', 'BXSL', 'BYAH', 'BYD', 'BYFC', 'BYM', 'BYSI', 'BZ', 'BZAIW', 'BZFD', 'BZFDW', 'BZUN', 'CA', 'CAAS', 'CABA', 'CACI', 'CAE', 'CAEP', 'CAF', 'CAI', 'CALC', 'CALY', 'CAM', 'CAMP', 'CAN', 'CANF', 'CANG', 'CAPL', 'CAPN', 'CAPNR', 'CAPS', 'CAPT', 'CAPTW', 'CAQUU', 'CAS', 'CASI', 'CATO', 'CBAT', 'CBC', 'CBIO', 'CBK', 'CBUS', 'CCAP', 'CCC', 'CCCC', 'CCD', 'CCEC', 'CCEL', 'CCG', 'CCGWW', 'CCHH', 'CCID', 'CCIF', 'CCII', 'CCIIU', 'CCIIW', 'CCIX', 'CCIXU', 'CCIXW', 'CCK', 'CCLD', 'CCLDO', 'CCM', 'CCNEP', 'CCO', 'CCTG', 'CCU', 'CCXI', 'CCXIU', 'CCXIW', 'CCZ', 'CD', 'CDIO', 'CDIOW', 'CDLR', 'CDLX', 'CDNL', 'CDRO', 'CDROW', 'CDT', 'CDTG', 'CDTTW', 'CDZIP', 'CEE', 'CELU', 'CELUW', 'CELZ', 'CENN', 'CEPF', 'CEPO', 'CEPS', 'CEPT', 'CEPV', 'CERT', 'CET', 'CETX', 'CETY', 'CEV', 'CFND', 'CGABL', 'CGAU', 'CGBD', 'CGC', 'CGCT', 'CGCTU', 'CGCTW', 'CGEN', 'CGNT', 'CGNX', 'CGO', 'CGRO', 'CGTL', 'CGTX', 'CGV', 'CHA', 'CHAC', 'CHACR', 'CHACU', 'CHAI', 'CHAR', 'CHARR', 'CHCI', 'CHEC', 'CHECU', 'CHECW', 'CHGG', 'CHH', 'CHI', 'CHMI', 'CHNR', 'CHOW', 'CHPG', 'CHPGR', 'CHPGU', 'CHR', 'CHSCL', 'CHSCM', 'CHSCN', 'CHSCO', 'CHSCP', 'CHSN', 'CHT', 'CHW', 'CHY', 'CHYM', 'CICB', 'CICC', 'CIF', 'CIG', 'CIG.C', 'CIGI', 'CIGL', 'CII', 'CIIT', 'CIK', 'CIMN', 'CIMO', 'CIMP', 'CING', 'CINGW', 'CINT', 'CION', 'CISO', 'CISS', 'CITR', 'CJMB', 'CKX', 'CLBT', 'CLDI', 'CLGN', 'CLIK', 'CLIR', 'CLLS', 'CLM', 'CLNN', 'CLPS', 'CLRB', 'CLRO', 'CLS', 'CLSKW', 'CLST', 'CLWT', 'CLYM', 'CM', 'CMBM', 'CMBT', 'CMCM', 'CMCT', 'CMIIU', 'CMMB', 'CMND', 'CMPS', 'CMS', 'CMSA', 'CMSC', 'CMSD', 'CMTL', 'CMTV', 'CMU', 'CNCK', 'CNCKW', 'CNET', 'CNEY', 'CNF', 'CNH', 'CNL', 'CNM', 'CNOBP', 'CNSP', 'CNTA', 'CNTB', 'CNTX', 'CNTY', 'CNVS', 'CNXC', 'COCH', 'COCHW', 'COCP', 'CODA', 'COE', 'COEP', 'COEPW', 'COHN', 'COLA', 'COLAR', 'COLAU', 'COLB', 'COLD', 'CONX', 'COOT', 'COOTW', 'COPL', 'COR', 'CORZW', 'CORZZ', 'COSM', 'COYA', 'CPA', 'CPAC', 'CPAY', 'CPB', 'CPBI', 'CPHC', 'CPHI', 'CPII', 'CPIX', 'CPOP', 'CPSH', 'CPSR', 'CPST', 'CPZ', 'CQP', 'CR', 'CRAC', 'CRACR', 'CRACU', 'CRACW', 'CRAN', 'CRANR', 'CRANU', 'CRAQ', 'CRAQR', 'CRAQU', 'CRBD', 'CRBG', 'CRBP', 'CRBU', 'CRCL', 'CRD.B', 'CRDL', 'CRE', 'CREG', 'CRESW', 'CRESY', 'CREX', 'CRF', 'CRGO', 'CRGOW', 'CRIS', 'CRMLW', 'CRNT', 'CRON', 'CRT', 'CRTO', 'CRVO', 'CRWS', 'CRWV', 'CSAI', 'CSAN', 'CSBR', 'CSGP', 'CSIQ', 'CSQ', 'CSRE', 'CSTE', 'CSWC', 'CTBB', 'CTDD', 'CTEC', 'CTM', 'CTMX', 'CTNM', 'CTNT', 'CTOR', 'CTRM', 'CTSO', 'CTW', 'CTXR', 'CUB', 'CUBB', 'CUBWU', 'CUBWW', 'CUE', 'CUK', 'CULP', 'CUPR', 'CURR', 'CURX', 'CUZ', 'CV', 'CVE', 'CVEO', 'CVGI', 'CVKD', 'CVM', 'CVR', 'CVU', 'CVV', 'CWD', 'CWEN.A', 'CX', 'CXAI', 'CXAIW', 'CXE', 'CXH', 'CXT', 'CYCN', 'CYCU', 'CYCUW', 'CYD', 'CYN', 'CYPH', 'DAAQ', 'DAAQU', 'DAAQW', 'DAC', 'DAIC', 'DAICW', 'DAIO', 'DAR', 'DARE', 'DAVA', 'DAVEW', 'DB', 'DBCAU', 'DBGI', 'DBL', 'DBVT', 'DCH', 'DCI', 'DCOMG', 'DCOMP', 'DCOY', 'DCX', 'DDC', 'DDI', 'DDL', 'DDS', 'DDT', 'DEFT', 'DEMZ', 'DEVS', 'DFDV', 'DFDVW', 'DFLI', 'DFLIW', 'DFNS', 'DFNSW', 'DFP', 'DFSC', 'DFSCW', 'DFTX', 'DGICB', 'DGNX', 'DGXX', 'DHCNI', 'DHCNL', 'DHF', 'DHX', 'DHY', 'DIAX', 'DIBS', 'DIT', 'DJT', 'DJTWW', 'DKI', 'DKL', 'DLHC', 'DLNG', 'DLPN', 'DLTH', 'DLXY', 'DLY', 'DMA', 'DMAA', 'DMAAR', 'DMAAU', 'DMB', 'DMII', 'DMIIR', 'DMIIU', 'DMLP', 'DMO', 'DNMX', 'DNMXU', 'DNMXW', 'DNP', 'DOGZ', 'DOMH', 'DOO', 'DOV', 'DPG', 'DPRO', 'DQ', 'DRAY', 'DRCT', 'DRD', 'DRDB', 'DRDBW', 'DRIO', 'DRMA', 'DRMAW', 'DRS', 'DRTS', 'DRTSW', 'DSAC', 'DSACU', 'DSACW', 'DSCO', 'DSGX', 'DSL', 'DSM', 'DSS', 'DSU', 'DSWL', 'DSX', 'DSY', 'DSYWW', 'DTB', 'DTCK', 'DTCX', 'DTE', 'DTF', 'DTG', 'DTI', 'DTIL', 'DTK', 'DTSQ', 'DTSQR', 'DTSS', 'DTST', 'DTSTW', 'DTW', 'DUKB', 'DUKH', 'DUO', 'DUOT', 'DVLT', 'DVLU', 'DVOL', 'DVS', 'DWAW', 'DWSN', 'DWTX', 'DWUS', 'DXF', 'DXLG', 'DXR', 'DXST', 'DXYZ', 'DYAI', 'DYOR', 'DYORU', 'DYORW', 'E', 'EAD', 'EAF', 'EAGL', 'EAI', 'EARN', 'EBIZ', 'EBON', 'ECAT', 'ECC', 'ECCC', 'ECCU', 'ECCV', 'ECCW', 'ECCX', 'ECF', 'ECG', 'ECO', 'ECOR', 'ECX', 'ECXWW', 'EDAP', 'EDBL', 'EDBLW', 'EDD', 'EDF', 'EDGE', 'EDHL', 'EDN', 'EDRY', 'EDSA', 'EDTK', 'EDUC', 'EEA', 'EEE', 'EEFT', 'EEIQ', 'EFOI', 'EFR', 'EFSCP', 'EFT', 'EFX', 'EFXT', 'EGG', 'EGHA', 'EGHAR', 'EGLE', 'EGO', 'EGP', 'EH', 'EHGO', 'EHI', 'EHLD', 'EIC', 'EICA', 'EICC', 'EIIA', 'EIKN', 'EIM', 'EJH', 'EKSO', 'ELAB', 'ELBM', 'ELC', 'ELE', 'ELLO', 'ELOG', 'ELPC', 'ELPW', 'ELSE', 'ELTK', 'ELTX', 'ELUT', 'ELVA', 'ELVR', 'ELWT', 'EM', 'EMA', 'EMAT', 'EMBJ', 'EMC', 'EMD', 'EMES', 'EMF', 'EMIS', 'EMISR', 'EMO', 'EMP', 'EMPD', 'EMXF', 'ENGN', 'ENGNW', 'ENGS', 'ENIC', 'ENJ', 'ENLT', 'ENLV', 'ENO', 'ENSC', 'ENTX', 'ENVB', 'EOD', 'EOI', 'EONR', 'EOS', 'EOT', 'EPOW', 'EPR', 'EPRX', 'EPSM', 'EQ', 'EQPT', 'EQS', 'EQX', 'ERC', 'ERH', 'ERNA', 'ERNAW', 'ESAB', 'ESBA', 'ESEA', 'ESGL', 'ESGLW', 'ESHA', 'ESHAR', 'ESI', 'ESLA', 'ESLAW', 'ESLT', 'ESP', 'ESSC', 'ESTA', 'ETB', 'ETG', 'ETH', 'ETHA', 'ETHM', 'ETHMU', 'ETHMW', 'ETHZ', 'ETJ', 'ETO', 'ETOR', 'ETS', 'ETV', 'ETW', 'ETX', 'ETY', 'EUDA', 'EUDAW', 'EURK', 'EURKR', 'EVAC', 'EVAX', 'EVF', 'EVG', 'EVGN', 'EVGOW', 'EVLVW', 'EVMN', 'EVN', 'EVO', 'EVOX', 'EVOXU', 'EVOXW', 'EVT', 'EVTL', 'EVTV', 'EVV', 'EWBC', 'EXE', 'EXEL', 'EXG', 'EXK', 'EXOD', 'EXOZ', 'EYEG', 'EZGO', 'EZPW', 'EZRA', 'FACT', 'FACTU', 'FACTW', 'FAF', 'FAMI', 'FARM', 'FATN', 'FAX', 'FB', 'FBGL', 'FBIN', 'FBIO', 'FBIOP', 'FBLG', 'FBRX', 'FBYD', 'FBYDW', 'FCEL', 'FCHL', 'FCN', 'FCNCA', 'FCNCN', 'FCNCO', 'FCNCP', 'FCO', 'FCRS', 'FCRX', 'FCT', 'FCUV', 'FDSB', 'FDUS', 'FEAC', 'FEAM', 'FEBO', 'FEDU', 'FEED', 'FEMY', 'FENG', 'FER', 'FERA', 'FERAR', 'FERAU', 'FFA', 'FFAIW', 'FFC', 'FFIV', 'FGBI', 'FGBIP', 'FGI', 'FGIIU', 'FGIWW', 'FGL', 'FGMC', 'FGMCR', 'FGMCU', 'FGN', 'FGNX', 'FGNXP', 'FGSN', 'FHB', 'FHI', 'FICS', 'FIEE', 'FIG', 'FIGR', 'FIGX', 'FIGXU', 'FIGXW', 'FINS', 'FINV', 'FITBI', 'FITBM', 'FITBO', 'FITBP', 'FJET', 'FKWL', 'FLC', 'FLD', 'FLDDW', 'FLEX', 'FLL', 'FLNT', 'FLO', 'FLOW', 'FLS', 'FLUT', 'FLUX', 'FLX', 'FLXN', 'FLY', 'FLYE', 'FLYX', 'FMFC', 'FMN', 'FMS', 'FMST', 'FMSTW', 'FMX', 'FMY', 'FNB', 'FNGR', 'FNWB', 'FOF', 'FOFO', 'FONR', 'FORA', 'FORTY', 'FOSL', 'FOXX', 'FOXXW', 'FPF', 'FPH', 'FPS', 'FRA', 'FRGT', 'FRHC', 'FRMEP', 'FRMI', 'FRO', 'FROG', 'FRPT', 'FRSX', 'FSCO', 'FSEA', 'FSHP', 'FSHPR', 'FSHPU', 'FSI', 'FSK', 'FSM', 'FSSL', 'FSV', 'FT', 'FTAI', 'FTAIM', 'FTAIN', 'FTCI', 'FTEK', 'FTEL', 'FTF', 'FTFT', 'FTHM', 'FTHY', 'FTI', 'FTPA', 'FTRK', 'FTS', 'FTW', 'FUFU', 'FUFUW', 'FULTP', 'FUND', 'FURY', 'FUSB', 'FUSE', 'FUSEW', 'FUTU', 'FVN', 'FVNNR', 'FVNNU', 'FVRR', 'FWDI', 'FWONA', 'FWONK', 'G', 'GAB', 'GAIN', 'GAINI', 'GAINN', 'GAINZ', 'GALT', 'GAM', 'GAME', 'GANX', 'GAP', 'GASS', 'GAU', 'GAUZ', 'GBAB', 'GBDC', 'GBLI', 'GBR', 'GCDT', 'GCL', 'GCLWW', 'GCTK', 'GCTS', 'GCV', 'GDC', 'GDEV', 'GDEVW', 'GDHG', 'GDL', 'GDO', 'GDRX', 'GDTC', 'GDV', 'GECC', 'GECCG', 'GECCH', 'GECCI', 'GECCO', 'GEG', 'GEGGL', 'GELS', 'GEMI', 'GENK', 'GENT', 'GENVR', 'GEOS', 'GES', 'GF', 'GFAI', 'GFAIW', 'GFI', 'GFR', 'GGM', 'GGN', 'GGR', 'GGROW', 'GGT', 'GGZ', 'GHG', 'GHI', 'GHRS', 'GHY', 'GIB', 'GIBO', 'GIBOW', 'GIFT', 'GIG', 'GIGGU', 'GIGGW', 'GIGM', 'GILT', 'GIPR', 'GIPRW', 'GITS', 'GIW', 'GIWWR', 'GIWWU', 'GIXXU', 'GJH', 'GJO', 'GJP', 'GJR', 'GJS', 'GJT', 'GK', 'GL', 'GLAD', 'GLBL', 'GLBS', 'GLDG', 'GLE', 'GLIBA', 'GLIBK', 'GLMD', 'GLO', 'GLOO', 'GLOW', 'GLP', 'GLPG', 'GLQ', 'GLTO', 'GLU', 'GLV', 'GLXG', 'GLXY', 'GMHS', 'GMM', 'GNLN', 'GNLX', 'GNOM', 'GNPX', 'GNS', 'GNSS', 'GNT', 'GNTA', 'GOAI', 'GOF', 'GOODN', 'GOODO', 'GOOS', 'GORO', 'GOTU', 'GOVX', 'GP', 'GPAC', 'GPACU', 'GPACW', 'GPAT', 'GPATW', 'GPCR', 'GPGI', 'GPJA', 'GPMT', 'GPRO', 'GPT', 'GPUS', 'GRABW', 'GRAF', 'GRAF.U', 'GRAN', 'GRCE', 'GRDX', 'GREE', 'GREEL', 'GRF', 'GRFS', 'GRI', 'GRIN', 'GRNQ', 'GRO', 'GROV', 'GROW', 'GROY', 'GRRR', 'GRRRW', 'GRVY', 'GRX', 'GSBD', 'GSHR', 'GSHRW', 'GSIG', 'GSIT', 'GSIW', 'GSL', 'GSOL', 'GSRF', 'GSRFR', 'GSRFU', 'GSUN', 'GSX', 'GTBP', 'GTE', 'GTEC', 'GTEN', 'GTENU', 'GTENW', 'GTERA', 'GTERR', 'GTERU', 'GTERW', 'GTIM', 'GTM', 'GTN.A', 'GUG', 'GURE', 'GUT', 'GUTS', 'GV', 'GVH', 'GWAV', 'GWH', 'GXAI', 'GYRO', 'HACQU', 'HAFN', 'HAO', 'HAVA', 'HAVAR', 'HAVAU', 'HAYW', 'HBANL', 'HBANM', 'HBANP', 'HBANZ', 'HBIO', 'HBNB', 'HBR', 'HBTA', 'HCAC', 'HCACR', 'HCACU', 'HCAI', 'HCHL', 'HCICU', 'HCM', 'HCMA', 'HCMAU', 'HCMAW', 'HCTI', 'HCWB', 'HCWC', 'HCXY', 'HDL', 'HELP', 'HEPS', 'HEQ', 'HERE', 'HERO', 'HERZ', 'HESM', 'HF', 'HFBL', 'HFRO', 'HGBL', 'HGLB', 'HGTY', 'HHH', 'HHS', 'HIHO', 'HIMX', 'HIND', 'HIO', 'HIT', 'HITI', 'HIVE', 'HIW', 'HIX', 'HKD', 'HKIT', 'HKPD', 'HLI', 'HLNE', 'HLP', 'HLXC', 'HMR', 'HNGE', 'HNNA', 'HNNAZ', 'HOFT', 'HOLO', 'HOLOW', 'HOTH', 'HOUR', 'HOVNP', 'HOVR', 'HOVRW', 'HOWL', 'HPAI', 'HPAIW', 'HPF', 'HPI', 'HPS', 'HQH', 'HQL', 'HRB', 'HRZN', 'HSAI', 'HSCS', 'HSCSW', 'HSDT', 'HSPT', 'HSPTR', 'HSPTU', 'HTCO', 'HTCR', 'HTD', 'HTFC', 'HTFL', 'HTGC', 'HTLM', 'HTOO', 'HTT', 'HTZWW', 'HUBC', 'HUBCW', 'HUBCZ', 'HUDI', 'HUHU', 'HUIZ', 'HUMAW', 'HUN', 'HURC', 'HVII', 'HVIIR', 'HVIIU', 'HVMC', 'HVMCU', 'HVMCW', 'HVT.A', 'HWAY', 'HWCPZ', 'HWH', 'HXHX', 'HYAC', 'HYFM', 'HYFT', 'HYI', 'HYMC', 'HYNE', 'HYPD', 'HYPR', 'HYT', 'IACOU', 'IAE', 'IAF', 'IAG', 'IBAC', 'IBACR', 'IBCA', 'IBG', 'IBIO', 'IBO', 'IBTG', 'IBTH', 'IBTI', 'IBTJ', 'ICCC', 'ICCM', 'ICG', 'ICL', 'ICMB', 'ICON', 'ICU', 'ICUCW', 'IDA', 'IDAI', 'IDE', 'IDN', 'IEAGU', 'IEP', 'IEUS', 'IFBD', 'IFGL', 'IFLO', 'IFN', 'IFRX', 'IFS', 'IGA', 'IGAC', 'IGACR', 'IGACU', 'IGC', 'IGD', 'IGF', 'IGI', 'IGIC', 'IGLD', 'IGR', 'IH', 'IHD', 'IHG', 'IHS', 'IHT', 'IHYF', 'IIF', 'IIM', 'IINN', 'IINNW', 'ILAG', 'IMA', 'IMCB', 'IMCC', 'IMDX', 'IMF', 'IMG', 'IMMP', 'IMMX', 'IMNN', 'IMO', 'IMOS', 'IMPP', 'IMPPP', 'IMRA', 'IMRN', 'IMRX', 'IMSR', 'IMSRW', 'IMTE', 'IMTX', 'IMUX', 'INAB', 'INAC', 'INACR', 'INACU', 'INBKZ', 'INBS', 'INCR', 'IND', 'INDO', 'INDP', 'INEO', 'INFO', 'INFU', 'INGM', 'INGR', 'INHD', 'INKT', 'INLF', 'INLX', 'INM', 'INO', 'INOV', 'INSM', 'INTG', 'INTJ', 'INTL', 'INTR', 'INTS', 'INTT', 'INTZ', 'INUV', 'INV', 'INVE', 'INVN', 'INVZ', 'INVZW', 'IOBT', 'IONR', 'IOR', 'IOTR', 'IPB', 'IPCX', 'IPCXR', 'IPCXU', 'IPDN', 'IPEX', 'IPEXR', 'IPEXU', 'IPHA', 'IPM', 'IPOD', 'IPODU', 'IPODW', 'IPSC', 'IPST', 'IPW', 'IPWR', 'IPX', 'IQI', 'IQM', 'IQST', 'IRD', 'IRE', 'IREN', 'IRET', 'IRHO', 'IRHOR', 'IRHOU', 'IRIX', 'IROQ', 'IRS', 'ISBA', 'ISD', 'ISOU', 'ISPC', 'ISSC', 'ISTB', 'ITHA', 'ITHAU', 'ITHAW', 'ITOC', 'ITP', 'ITRG', 'ITRM', 'ITRN', 'ITWO', 'IVA', 'IVDA', 'IVDAW', 'IVF', 'IVVD', 'IVZ', 'IX', 'IXHL', 'IZEA', 'IZM', 'JACS', 'JADE', 'JAGU', 'JAGX', 'JAVA', 'JBDI', 'JBK', 'JBS', 'JCE', 'JCSE', 'JCTC', 'JDZG', 'JEF', 'JEM', 'JENA', 'JF', 'JFB', 'JFBR', 'JFBRW', 'JFIN', 'JFR', 'JFU', 'JG', 'JGH', 'JHG', 'JHI', 'JHS', 'JHX', 'JIVE', 'JKS', 'JL', 'JLHL', 'JLS', 'JMIA', 'JMM', 'JOB', 'JOET', 'JOF', 'JOYY', 'JPC', 'JQC', 'JRI', 'JRS', 'JRSH', 'JSM', 'JSPR', 'JSPRW', 'JTAI', 'JUNS', 'JVA', 'JWEL', 'JXG', 'JYD', 'JZ', 'JZXN', 'KALA', 'KAPA', 'KARO', 'KB', 
 
   

]

PERIODO_ANOS_ENTRENAMIENTO = 3          # años de historia para cada loop
FWD_DAYS = 20                           # días hacia adelante p/target de entrenamiento (clasificador)
FWD_DAYS_TARGET = 7                     # días hacia adelante p/retorno “una semana”
N_SEMANAS_BACKTEST = 2                 # ← por ahora 5; si exporta ok, subimos a 100
OUTPUT_XLSX = Path("Consolidado_1_semanas_todos los tickers_2.xlsx")
SEED = 42

# ----------------- FEATURES -----------------
FEATURES_GEMINI = [
    'MFI_14','MACDh_12_26_9','PctB','CCI_20_0.015','OBV_trend',
    'STOCH_Cross','ADX_14','MA_SLOPE_20','Cross_Signal','ATRp_14',
    'VROC_14','Consecutive_Volume_Growth','FIB_Range_90D',
]
FEATURES_CGPT = [
    'RSI_14','MACD_Hist_Slope_5','RET_1M','RET_3M','RET_6M',
    'Aroon_Diff_25','Vol_Rel_20','Vol_StreakUp','Days_since_Golden','Days_since_Death',
    'Ret13_Ratio','Pct_in_52w_range','Drawdown_52w','SMA_50','SMA_200'
]
FEATURES_MODEL_BASE = list(sorted(set(FEATURES_GEMINI + FEATURES_CGPT)))  # RS*_SPY se agregan luego

# =============== HELPERS: indicadores ===============
def _streak_up(s: pd.Series) -> pd.Series:
    inc = (s > s.shift(1)).astype(int)
    run, out = 0, []
    for v in inc.fillna(0):
        run = run + 1 if v == 1 else 0
        out.append(run)
    return pd.Series(out, index=s.index)

def _days_since_cross(short: pd.Series, long: pd.Series, cross='golden') -> pd.Series:
    short = short.copy(); long = long.copy()
    above_prev = (short.shift(1) > long.shift(1))
    above_now  = (short > long)
    if cross == 'golden':
        evt = (~above_prev.fillna(False)) & (above_now.fillna(False))
    else:
        evt = (above_prev.fillna(False)) & (~above_now.fillna(False))

    res = pd.Series(np.nan, index=short.index)
    last_idx = None
    for i, flag in enumerate(evt):
        if flag:
            last_idx = i
        if last_idx is not None:
            res.iat[i] = i - last_idx

    if res.notna().any():
        res = res.ffill()
    else:
        # Nunca ocurrió el cruce en el rango: usar un valor grande
        res[:] = len(res) + 1
    return res


def _aroon(df: pd.DataFrame, period=25):
    high = df['High']; low = df['Low']
    hh_idx = high.rolling(period).apply(np.argmax, raw=True)
    ll_idx = low.rolling(period).apply(np.argmin, raw=True)
    up = 100 * (period - 1 - hh_idx) / (period - 1)
    down = 100 * (period - 1 - ll_idx) / (period - 1)
    return up, down, up - down

def add_all_indicators(df):
    """Calcula todos los indicadores (Gemini + ChatGPT) en 'df' con columnas OHLCV."""
    if df.empty or len(df) < 100:
        return None
    # Asegurar nombres
    df = df.rename(columns={c: c.capitalize() for c in df.columns})
    # --- Gemini ---
    df.ta.mfi(length=14, append=True)                               # MFI_14
    _ = df.ta.macd(append=True)                                     # MACD_12_26_9, MACDh_12_26_9
    bb = df.ta.bbands(length=20, std=2, append=False)               # BBP
    if bb is not None and not bb.empty:
        bbp_cols = [c for c in bb.columns if "BBP" in c]
        df['PctB'] = bb[bbp_cols[0]] if bbp_cols else np.nan
    else:
        df['PctB'] = np.nan
    df.ta.cci(length=20, append=True)                               # CCI_20_0.015
    df.ta.obv(append=True)                                          # OBV
    stoch = df.ta.stoch(k=14, d=3, smooth_k=3)
    if stoch is not None:
        df = pd.concat([df, stoch], axis=1)
    df.ta.adx(length=14, append=True)                               # ADX_14
    df.ta.sma(length=20, append=True)                               # SMA_20
    df.ta.sma(length=50, append=True)                               # SMA_50
    df.ta.sma(length=200, append=True)                              # SMA_200

    # ✅ Fallbacks por si pandas_ta no generó las columnas
    if 'SMA_50' not in df.columns:
        df['SMA_50'] = df['Close'].rolling(50, min_periods=1).mean()
    if 'SMA_200' not in df.columns:
        df['SMA_200'] = df['Close'].rolling(200, min_periods=1).mean()

        # --- ATR & ATR% robusto ---
    _ = df.ta.atr(length=14, append=True)  # agrega alguna de: ATR_14, ATRr_14, ATRs_14...
    atr_col = next((c for c in df.columns if c.upper().startswith('ATR') and c.endswith('_14')), None)

    if atr_col is None:
        # Cálculo manual de TR y ATR(14) si por algún motivo no quedó columna
        tr1 = df['High'] - df['Low']
        tr2 = (df['High'] - df['Close'].shift(1)).abs()
        tr3 = (df['Low']  - df['Close'].shift(1)).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr14 = tr.rolling(14, min_periods=1).mean()
    else:
        atr14 = df[atr_col]

    df['ATRp_14'] = (atr14 / df['Close']).replace([np.inf, -np.inf], np.nan) * 100
    


    # VROC (volume ROC manual, %)
    df['VROC_14'] = df['Volume'].pct_change(14) * 100
    df['OBV_trend'] = df.get('OBV', pd.Series(index=df.index)).pct_change(20).replace([np.inf, -np.inf], np.nan)
    if 'STOCHk_14_3_3' in df.columns and 'STOCHd_14_3_3' in df.columns:
        k, d = df['STOCHk_14_3_3'], df['STOCHd_14_3_3']
        cross_up = (k.shift(1) <= d.shift(1)) & (k > d)
        cross_dn = (k.shift(1) >= d.shift(1)) & (k < d)
        df['STOCH_Cross'] = np.select([cross_up, cross_dn], [1, -1], default=0)
    else:
        df['STOCH_Cross'] = 0
    if 'SMA_20' in df.columns:
        df['MA_SLOPE_20'] = df['SMA_20'].diff(5) / df['SMA_20'].shift(5).replace(0, np.nan)
    else:
        df['MA_SLOPE_20'] = np.nan
    if 'SMA_50' in df.columns and 'SMA_200' in df.columns:
        s50, s200 = df['SMA_50'], df['SMA_200']
        cross_g = (s50.shift(1) < s200.shift(1)) & (s50 > s200)
        cross_d = (s50.shift(1) > s200.shift(1)) & (s50 < s200)
        df['Cross_Signal'] = np.select([cross_g, cross_d], [1, -1], default=0)
    else:
        df['Cross_Signal'] = 0
    if 'Volume' in df.columns:
        vol_grows = df['Volume'] > df['Volume'].shift(1)
        df['Consecutive_Volume_Growth'] = vol_grows.cumsum() - vol_grows.cumsum().where(~vol_grows).ffill().fillna(0)
    else:
        df['Consecutive_Volume_Growth'] = 0
    roll_90 = df['Close'].rolling(90)
    swing_high = roll_90.max(); swing_low = roll_90.min()
    df['FIB_Range_90D'] = (df['Close'] - swing_low) / (swing_high - swing_low).replace(0, np.nan)

    # --- ChatGPT ---
    df.ta.rsi(length=14, append=True)                               # RSI_14
    if 'MACDh_12_26_9' in df.columns:
        df['MACD_Hist_Slope_5'] = df['MACDh_12_26_9'].diff(5)
    else:
        df['MACD_Hist_Slope_5'] = np.nan
    df['MACD'] = df.get('MACD_12_26_9', np.nan)

    _, _, a_diff = _aroon(df, 25)
    df['Aroon_Diff_25'] = a_diff
    df['RET_1M'] = df['Close'].pct_change(20)
    df['RET_3M'] = df['Close'].pct_change(60)
    df['RET_6M'] = df['Close'].pct_change(126)
    if 'Volume' in df.columns:
        df['Vol_Rel_20'] = df['Volume'] / df['Volume'].rolling(20).mean().replace(0, np.nan)
        df['Vol_StreakUp'] = _streak_up(df['Volume'])
    else:
        df['Vol_Rel_20'] = np.nan; df['Vol_StreakUp'] = np.nan

    df['Days_since_Golden'] = _days_since_cross(df.get('SMA_50', pd.Series(index=df.index)),
                                                df.get('SMA_200', pd.Series(index=df.index)), 'golden')
    df['Days_since_Death']  = _days_since_cross(df.get('SMA_50', pd.Series(index=df.index)),
                                                df.get('SMA_200', pd.Series(index=df.index)), 'death')
    eps = 1e-9
    df['Ret13_Ratio'] = df['RET_1M'] / (df['RET_3M'].replace(0, eps) + eps)
    high_52w = df['Close'].rolling(252, min_periods=60).max()
    low_52w  = df['Close'].rolling(252, min_periods=60).min()
    rng_52w  = (high_52w - low_52w)
    df['Pct_in_52w_range'] = (df['Close'] - low_52w) / rng_52w.replace(0, np.nan)
    df['Drawdown_52w'] = (high_52w - df['Close']) / high_52w.replace(0, np.nan)

    # Limpieza final
    df = df.replace([np.inf, -np.inf], np.nan).fillna(method='bfill').fillna(method='ffill')
    return df

# =============== Scores macro + tag ===============
def add_macro_filters(df):
    L = df.copy()

    # helpers para garantizar Series
    def S(col, default):
        if col in L.columns:
            s = L[col]
            return s if isinstance(s, pd.Series) else pd.Series(s, index=L.index)
        else:
            return pd.Series(default, index=L.index)

    def _between(s, lo, hi):
        s = s if isinstance(s, pd.Series) else pd.Series(s, index=L.index)
        return (s >= lo) & (s <= hi)

    SMA50 = S('SMA_50', L['Close'])
    SMA200 = S('SMA_200', L['Close'])
    RSI   = S('RSI_14', 50.0)
    MACD  = S('MACD', 0.0)
    MACD_SLP = S('MACD_Hist_Slope_5', 0.0)
    ADX14 = S('ADX_14', 0.0)
    AROON = S('Aroon_Diff_25', 0.0)
    VOLREL = S('Vol_Rel_20', 1.0)
    PCTB = S('PctB', 0.5)
    ATRP = S('ATRp_14', 0.0)
    PCT52 = S('Pct_in_52w_range', 0.5)
    DD52  = S('Drawdown_52w', 0.2)
    RS1 = S('RS1M_SPY', 0.0)
    RS3 = S('RS3M_SPY', 0.0)
    RS6 = S('RS6M_SPY', 0.0)
    RET13 = S('Ret13_Ratio', 1.0)

    L['MA50_over_MA200'] = (SMA50 / SMA200.replace(0, np.nan)).fillna(1.0)
    L['Close_over_MA200'] = (L['Close'] / SMA200.replace(0, np.nan)).fillna(1.0)
    L['Price_gt_MA200'] = (L['Close'] > SMA200).astype(int)

    trend_score = ((L['Price_gt_MA200']==1).astype(int) +
                   (L['MA50_over_MA200'] >= 1.02).astype(int) +
                   (L['Close_over_MA200'] >= 1.02).astype(int) +
                   (AROON >= 50).astype(int) +
                   _between(ADX14, 18, 35).astype(int)) / 5 * 100

    momentum_score = (_between(RSI, 58, 72).astype(int) +
                      (MACD > 0).astype(int) +
                      (MACD_SLP > 0).astype(int) +
                      ((VOLREL >= 1.20) | _between(PCTB, 0.80, 1.10)).astype(int)) / 4 * 100

    rs_1 = (RS1 >= 0.02)
    rs_3 = (RS3 >= 0.04)
    rs_6 = (RS6 >= 0.00)
    relstr_score = (rs_1.astype(int) + rs_3.astype(int) + rs_6.astype(int) +
                    (RET13 >= 1.10).astype(int)) / 4 * 100

    breakout_vol = (PCT52 >= 0.90) & (VOLREL >= 1.20) & (ADX14 >= 20)
    near_high_no_vol = (PCT52 >= 0.90) & (VOLREL < 1.20)
    value_recovery = (_between(PCT52, 0.20, 0.50) &
                      _between(DD52, 0.20, 0.40) &
                      (MACD_SLP > 0) & (RSI >= 45))
    low_vol = (ATRP <= 4.0)

    riskpos_score = (breakout_vol.astype(int) + value_recovery.astype(int) +
                     low_vol.astype(int) + (~near_high_no_vol).astype(int)) / 4 * 100

    L['Trend_Score'] = trend_score.round(0)
    L['Momentum_Score'] = momentum_score.round(0)
    L['RelStr_Score'] = relstr_score.round(0)
    L['RiskPos_Score'] = riskpos_score.round(0)

    # Tags
    tags = []
    for i in range(len(L)):
        t = []
        if bool(((L['Price_gt_MA200'].iat[i]==1) and (L['MA50_over_MA200'].iat[i] >= 1.00) and
                 (AROON.iat[i] >= 20) and
                 (18 <= ADX14.iat[i] <= 40) and
                 (45 <= RSI.iat[i] <= 55) and
                 (0.20 <= PCTB.iat[i] <= 0.50) and
                 (MACD_SLP.iat[i] > 0))):
            t.append('Trend+Pullback')
        if bool(breakout_vol.iat[i]): t.append('Breakout+Vol')
        if bool((rs_1.astype(int) + rs_3.astype(int) + rs_6.astype(int)).iat[i] >= 2): t.append('RS Leader')
        if bool(value_recovery.iat[i]): t.append('Value Recov')
        if bool(((28 <= RSI.iat[i] <= 40) and
                 (MACD_SLP.iat[i] > 0) and
                 (PCTB.iat[i] <= 0.20) and
                 (ADX14.iat[i] < 25))):
            t.append('Rebound')
        if bool(near_high_no_vol.iat[i]): t.append('NearHigh NO Vol')
        if not t:
            arr = [trend_score.iat[i], momentum_score.iat[i], relstr_score.iat[i], riskpos_score.iat[i]]
            t.append(['Trend','Momentum','RelStr','RiskPos'][int(np.argmax(arr))])
        tags.append(' | '.join(t))
    L['Setup_Tag'] = tags
    L['NearHigh_NoVol'] = near_high_no_vol.astype(int)
    return L

# =============== ScoreSimple (reglas del árbol) ===============
def compute_score_simple(df):
    # 0..6 con penalizaciones
    sc = np.zeros(len(df), dtype=int)
    atr = df.get('ATRp_14', pd.Series([np.nan]*len(df))).values
    rs1 = df.get('RS1M_SPY', pd.Series([0]*len(df))).values
    r1m = df.get('RET_1M', pd.Series([0]*len(df))).values
    r6m = df.get('RET_6M', pd.Series([0]*len(df))).values
    m50_200 = df.get('MA50_over_MA200', pd.Series([1]*len(df))).values
    pctb = df.get('PctB', pd.Series([0.5]*len(df))).values
    vrel = df.get('Vol_Rel_20', pd.Series([1.0]*len(df))).values

    sc += ( (atr >= 3) & (atr <= 5.5) ).astype(int) * 2
    sc += ( (atr > 5.5) & (atr <= 8) ).astype(int) * 1
    sc -= ( (atr < 1.8) ).astype(int) * 2
    sc += ( rs1 >= 0.10 ).astype(int) * 1
    sc += ( (atr <= 3) & (r6m <= -0.08) ).astype(int) * 1
    sc += ( (atr > 3) & (r1m > -0.08) ).astype(int) * 1
    sc -= ( m50_200 >= 1.18 ).astype(int) * 1
    sc -= ( (pctb >= 0.90) & (vrel < 1.20) ).astype(int) * 1
    sc = np.maximum(sc, 0)
    df['ScoreSimple'] = sc
    return df

# =============== Descarga de datos ===============
def download_data(tickers, start_date, end_date):
    data = yf.download(
        tickers,
        start=start_date.strftime('%Y-%m-%d'),
        end=end_date.strftime('%Y-%m-%d'),
        auto_adjust=True,
        group_by='ticker',
        threads=True
    )
    if hasattr(data.columns, "remove_unused_levels"):
        data.columns = data.columns.remove_unused_levels()
    return data

# =============== Dataset por loop (entrena + RS_SPY) ===============
def build_dataset(tickers, data_slice):
    all_rows = []
    data_by_ticker = {}
    last_dates = []

    # si viene multiindex: level0=ticker, level1=OHLCV
    for tk in tickers:
        try:
            df = data_slice[tk].dropna().copy()
        except Exception:
            continue
        df = df.rename(columns=str.title)
        if not {'Open','High','Low','Close','Volume'} <= set(df.columns):
            continue

        feat = add_all_indicators(df[['Open','High','Low','Close','Volume']].copy())
        if feat is None or feat.empty:
            continue
        feat['Target_Return'] = (feat['Close'].shift(-FWD_DAYS) / feat['Close']) - 1
        feat['Ticker'] = tk
        all_rows.append(feat)
        data_by_ticker[tk] = feat
        last_dates.append(feat.index.max())

    if not all_rows:
        return pd.DataFrame(), {}, None

    training_df = pd.concat(all_rows)

    # ---- RS vs SPY (sobre RET_1M/3M/6M por fecha) ----
    if 'SPY' in data_by_ticker:
        spy = training_df[training_df['Ticker']=='SPY'][['RET_1M','RET_3M','RET_6M']].copy()
        spy = spy.rename(columns={'RET_1M':'SPY_RET_1M','RET_3M':'SPY_RET_3M','RET_6M':'SPY_RET_6M'})
        training_df = training_df.join(spy, how='left')
        training_df['RS1M_SPY'] = training_df['RET_1M'] - training_df['SPY_RET_1M']
        training_df['RS3M_SPY'] = training_df['RET_3M'] - training_df['SPY_RET_3M']
        training_df['RS6M_SPY'] = training_df['RET_6M'] - training_df['SPY_RET_6M']
        rs_cols = ['RS1M_SPY','RS3M_SPY','RS6M_SPY']
        training_df[rs_cols] = training_df[rs_cols].fillna(0.0)

        # propagar RS_* a data_by_ticker (para predicción)
        for tk, df in data_by_ticker.items():
            if tk == 'SPY':
                for c in rs_cols: df[c] = 0.0
            else:
                df[rs_cols] = training_df[training_df['Ticker']==tk][rs_cols]
                df[rs_cols] = df[rs_cols].fillna(0.0)
    else:
        for tk, df in data_by_ticker.items():
            df['RS1M_SPY'] = 0.0; df['RS3M_SPY'] = 0.0; df['RS6M_SPY'] = 0.0

    # target clasificador: top/bottom 40% cross-seccional por fecha
    ranks = training_df.groupby(training_df.index)['Target_Return'].rank(pct=True)
    training_df['Return_Rank'] = ranks
    training_df['Target_Class'] = np.where(ranks >= 0.6, 1, np.where(ranks <= 0.4, 0, np.nan))
    training_df = training_df.dropna(subset=['Target_Class']).copy()
    training_df['Target_Class'] = training_df['Target_Class'].astype(int)

    common_last_date = max(last_dates) if last_dates else None
    return training_df, data_by_ticker, common_last_date

# =============== Entrenar modelo ===============
def train_model(training_df, features):
    X = training_df[features].replace([np.inf,-np.inf], np.nan).fillna(0.0)
    y = training_df['Target_Class'].astype(int).values
    if len(np.unique(y)) < 2:
        return None
    clf = RandomForestClassifier(
        n_estimators=200, max_depth=12, min_samples_leaf=8,
        class_weight="balanced_subsample", random_state=SEED, n_jobs=-1
    )
    clf.fit(X, y)
    return clf

# =============== Predecir en fecha ===============
def predict_on_date(data_by_ticker, model, features, target_date):
    out = []
    for tk, df in data_by_ticker.items():
        df = df[df.index <= target_date]
        if df.empty:
            continue
        row = df.iloc[-1]

        # ✅ Asegurar que TODAS las features existan en la fila (faltantes = NaN -> 0.0)
        x = pd.DataFrame([row], columns=features)
        x = x.replace([np.inf, -np.inf], np.nan).fillna(0.0)

        try:
            proba = model.predict_proba(x)[:, 1][0]
        except Exception:
            proba = float(model.predict(x)[0])

        rec = row.to_dict()
        rec['Ticker'] = tk
        rec['Predicted_Outperform_Prob'] = float(proba)
        rec['Data_Date'] = df.index[-1]
        out.append(rec)
    return pd.DataFrame(out)

# =============== Targets + retornos a 1 semana ===============
def add_future_returns_and_targets(df_conso, data_full):
    look = {}
    for tk in df_conso['Ticker'].unique():
        try:
            sub = data_full[tk][['Close']].dropna().copy()
            sub['RET_1M'] = sub['Close'].pct_change(20)
            sub['RET_3M'] = sub['Close'].pct_change(60)
            sub['RET_6M'] = sub['Close'].pct_change(126)
            look[tk] = sub
        except Exception:
            continue

    def _fwd_price(row):
        tk = row['Ticker']; d0 = pd.Timestamp(row['Data_Date'])
        d1 = d0 + pd.Timedelta(days=FWD_DAYS_TARGET)
        if tk not in look: return np.nan
        s = look[tk]['Close']
        s = s[s.index >= d1]
        return float(s.iloc[0]) if not s.empty else np.nan

    df_conso['Precio una semana'] = df_conso.apply(_fwd_price, axis=1)
    df_conso['Ret. Una sem'] = (df_conso['Precio una semana'] / df_conso['Close']) - 1
    df_conso['Ret. Una sem'] = df_conso['Ret. Una sem'].fillna(0.0)

    spy_ok = 'SPY' in look
    if spy_ok:
        spy = look['SPY'].copy()
        spy['Ret_1w'] = (spy['Close'].shift(-FWD_DAYS_TARGET) / spy['Close']) - 1
        spy_std = float(spy['Ret_1w'].std(skipna=True))
        spy_map = spy['Ret_1w'].to_dict()

        for col in ['RET_1M','RET_3M','RET_6M']:
            m = spy[col].to_dict()
            df_conso[f'SPY_{col}'] = df_conso['Data_Date'].map(m)

        for h in ['RET_1M','RET_3M','RET_6M']:
            df_conso[f'RS{h[4:]}_SPY'] = df_conso[h] - df_conso[f'SPY_{h}']
        df_conso[['RS1M_SPY','RS3M_SPY','RS6M_SPY']] = df_conso[['RS1M_SPY','RS3M_SPY','RS6M_SPY']].fillna(0.0)

        df_conso['SPY_Ret'] = df_conso['Data_Date'].map(spy_map).fillna(0.0)
        df_conso['Min'] = df_conso['SPY_Ret'] - 2*spy_std
        df_conso['Max'] = df_conso['SPY_Ret'] + 2*spy_std
        df_conso['Target'] = np.select(
            [df_conso['Ret. Una sem'] < df_conso['Min'], df_conso['Ret. Una sem'] > df_conso['Max']],
            [1, 3], default=2
        )
    else:
        df_conso[['RS1M_SPY','RS3M_SPY','RS6M_SPY']] = 0.0
        df_conso['SPY_Ret'] = 0.0
        df_conso['Min'] = -0.05
        df_conso['Max'] = 0.05
        df_conso['Target'] = 2
    df_conso['Target3'] = (df_conso['Target'] == 3).astype(int)
    
    
    # (Opcional) Ocultar la semana más reciente para la que no hay +7 días aún
# (evita NaN en 'Precio una semana' / 'Ret. Una sem' del último viernes)
    try:
        last_trade_global = max(look[tk].index.max() for tk in look.keys())
        incomplete_cut = last_trade_global - pd.Timedelta(days=FWD_DAYS_TARGET)
        mask_incomplete = df_conso['Data_Date'] > incomplete_cut
        cols_to_null = ['Precio una semana', 'Ret. Una sem', 'Target', 'Target3', 'SPY_Ret', 'Min', 'Max']
        df_conso.loc[mask_incomplete, cols_to_null] = np.nan
    except Exception:
        pass

    
    
    return df_conso

# =============== MAIN ===============
def main():
    np.random.seed(SEED)

    # ------- 1) Descarga total -------
    dias_descarga = int((PERIODO_ANOS_ENTRENAMIENTO + (N_SEMANAS_BACKTEST/52)) * 365) + 252 + FWD_DAYS_TARGET
    end_date_total = datetime.now() + timedelta(days=1)
    start_date_total = end_date_total - timedelta(days=dias_descarga)
    print(f"Descargando {len(TICKERS)} tickers | {start_date_total.date()} → {end_date_total.date()}")
    data_full = download_data(TICKERS, start_date_total, end_date_total)
    if data_full is None or data_full.empty:
        print("❌ No se pudieron descargar datos.")
        return

    # ------- 2) Fechas semanales (últimos N viernes) -------
    backtest_dates = pd.date_range(end=datetime.now(), periods=N_SEMANAS_BACKTEST, freq='W-FRI').sort_values()

    all_rows = []
    features_model = FEATURES_MODEL_BASE.copy()  # se completan con RS_* al construir cada loop

    for loop_date in backtest_dates:
        loop_end = loop_date
        loop_start = loop_end - timedelta(days=int(PERIODO_ANOS_ENTRENAMIENTO*365) + 252 + FWD_DAYS)
        try:
            data_slice = data_full.loc[loop_start:loop_end].copy()
        except Exception:
            continue
        if data_slice.empty:
            continue

        training_df, data_by_ticker, last_date = build_dataset(TICKERS, data_slice)
        if training_df.empty or not data_by_ticker or last_date is None:
            continue

        # asegurar RS_* en features
        for c in ['RS1M_SPY','RS3M_SPY','RS6M_SPY']:
            if c not in features_model: features_model.append(c)

        # limpiar NaNs
        use_cols = [c for c in features_model if c in training_df.columns]
        training_df[use_cols] = training_df[use_cols].replace([np.inf,-np.inf], np.nan).fillna(0.0)
        model = train_model(training_df, use_cols)
        if model is None:
            continue

        pred_df = predict_on_date(data_by_ticker, model, use_cols, last_date)
        if pred_df.empty:
            continue

        keep = ['Ticker','Predicted_Outperform_Prob','Close','Data_Date'] + list(sorted(set(use_cols)))
        pred_df = pred_df[keep]
        all_rows.append(pred_df)

        print(f"Semana {last_date.date()} OK | filas: {len(pred_df)}")

    if not all_rows:
        print("❌ No hubo resultados.")
        return

    # ------- 3) Consolidado -------
    df_conso = pd.concat(all_rows, ignore_index=True)
    df_conso = df_conso.drop_duplicates(subset=['Ticker','Data_Date'])
    for h in ['RET_1M','RET_3M','RET_6M']:
        if h not in df_conso.columns:
            df_conso[h] = np.nan

    # ------- 4) Targets + RS con data_full -------
    df_conso = add_future_returns_and_targets(df_conso, data_full)

    # ------- 5) Scores macro + Setup_Tag -------
    df_conso = add_macro_filters(df_conso)

    # ------- 6) ScoreSimple (reglas del árbol) -------
    df_conso = compute_score_simple(df_conso)

    # ------- 7) Salida única -------
    df_conso = df_conso.sort_values(['Data_Date','Predicted_Outperform_Prob'], ascending=[True, False])

    if not pd.api.types.is_datetime64_any_dtype(df_conso['Data_Date']):
        df_conso['Data_Date'] = pd.to_datetime(df_conso['Data_Date'])

    with pd.ExcelWriter(OUTPUT_XLSX, engine='openpyxl') as writer:
        df_conso.to_excel(writer, sheet_name='Consolidado_5s', index=False)

    print(f"✅ Listo: {OUTPUT_XLSX.resolve()} | Filas: {len(df_conso)} | Tickers únicos: {df_conso['Ticker'].nunique()}")

if __name__ == "__main__":
    main()


