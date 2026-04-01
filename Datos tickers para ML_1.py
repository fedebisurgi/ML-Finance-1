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
   'OKLO','AVAV','BBIO','RNA','ENSG','STRL','SPXC','DY','WTS','HQY','AHR','SANM','IDCC','QBTS','CTRE','CYTK','NUVL','ZWS','AEIS','MOD','AXSM','IESC','STEP','CRK','JBTM','FCFS','RYTM','CDTX','CADE','TTMI','IBP','WAY','PRIM','ORA','CDP','FSS','APLD','PFSI','PCVX','TXNM','MOG.A','JXN','SGHC','RDNT','MIR','PTCT','CWAN','ROAD','RRR','BTSG','CWST','VLY','RHP','CIFR','SWX','PIPR','ESE','GKOS','ACT','IRTC','ARWR','BOOT','COMP','PTGX','LIF','BKH','HOMB','COGT','KNTK','BIPC','GBCI','CVLT','UFPI','WULF','ACLX','SR','WK','AGX','BMI','BCPC','QLYS','MMSI','MC','ACA','ABCB','OGS','MCY','SOUN','PECO','TGTX','ACIW','NE','HWC','ALKS','MZTI','CORZ','SBRA','KYMR','SIGI','NJR','GHC','GOLF','MTH','AUB','BCO','TMDX','YOU','NNI','RIG','FLG','LAUR','PI','MMS','NPO','APGE','AX','PBF','CELC','SSRM','FFIN','REZI','SKY','BDC','ABG','MGY','SNEX','MAC','CVCO','CALM','HASI','EBC','CRNX','OSIS','TDS','ZETA','MRCY','FG','ASB','AROC','GPOR','SRRK','LGND','IMVT','CSW','NWE','INDV','FELE','BGC','CNR','HCC','IRT','ZGN','VSEC','IBOC','SXT','HRI','RSI','CRC','KNF','FORM','ALE','ACAD','PRM','TCBI','GLNG','WGS','ATMU','SKT','VICR','CNO','COMM','RUSHA','SFBS','KBH','RUSHB','ADMA','SKYW','NHI','PLXS','OUT','CRVL','EOSE','CALX','FTDR','AMRX','GBTG','VIAV','MWA','MIRM','BANF','CVI','UCB','NOVT','PSMT','CNK','HAE','VRRM','PII','LNTH','ATGE','CWK','AGYS','BNL','GRAL','IRON','OTTR','ARQT','CGON','EXPO','USLM','AHL','MYRG','GNW','BTU','MLYS','POWL','PLMR','KFY','AVA','APAM','FRSH','INDB','ICUI','ANF','PATK','TARS','ALHC','CARG','JOE','MHO','MH','WHD','TVTX','CPK','HGV','GFF','RNST','BL','BWIN','VCYT','BBUC','GTX','CBT','FIBK','XENE','VERX','FULT','FBP','STNG','FIZZ','XMTR','SEI','CNS','EE','CLSK','FUL','BRZE','ENVA','DYN','FOLD','GEF.B','AZZ','AEO','MGEE','LIVN','PCH','ATEC','PPTA','WSFS','BRSL','KAI','IDYA','WSBC','KGS','MCHB','SBCF','LRN','CBU','TEX','FBK','NMRK','BKE','BFH','HURN','SXI','PRVA','SPHR','TDW','AVPT','VRDN','LXP','TR','BLKB','CBZ','APLE','TTAM','LBRT','COCO','GRBK','NSIT','KAR','RELY','HP','AGIO','WOR','INSW','TPH','NATL','CENX','BANC','HWKN','TOWN','AKR','WDFC','TBBK','PACS','DAVE','LQDA','QUBT','OCUL','LCII','FCPT','SLNO','GRND','CVBF','HG','BBAI','SFNC','GSHD','RCUS','DNLI','BKD','REVG','BCC','ABM','CON','TERN','WAFD','NUVB','CURB','DK','PRK','LFST','BATRK','SPNS','BATRA','ALRM','UPWK','CMPO','MTRN','PLUS','MGRC','RUM','PAX','PFS','EWTX','SYNA','CHEF','OII','MRX','BTDR','PINC','FFBC','DNOW','ADPT','SPNT','CRGY','HBI','EXTR','TRMK','AVDL','ATKR','DRVN','DBD','GENI','HI','CUBI','AMBP','KWR','PGNY','CAKE','ALKT','IVT','GCMG','NOG','TGLS','UNFI','IBRX','DHT','FA','WD','TNK','AORT','NBTB','AMR','CSTM','SGRY','BBT','GEO','SMA','ACMR','BANR','FRME','FBNC','PHIN','OSW','TIC','TRN','ADUS','HTH','STC','LION','AUPH','EPAC','ZBIO','BUSE','LMAT','SLDE','DX','MD','EYE','FIHL','OI','NWN','NHC','IMAX','EFSC','TPB','KMT','HE','SMPL','HRMY','SDRL','ALG','SYBT','SKWD','PRDO','SYRE','IE','ARR','BUR','NTB','EVTC','HMN','KLIC','STRA','CECO','BELFA','DRH','AI','VERA','TALO','BELFB','SCS','KN','RXO','RAMP','ANIP','HNI','SEZL','MTX','CMRE','DBRG','SLVM','NIC','WLY','PGY','INOD','ATRO','CENT','CTRI','AGM','AVAH','CENTA','IOSP','SEMR','GRDN','JJSF','NGVT','DNTH','OFG','GNL','LTC','MCRI','SION','SHO','JANX','MODG','MAZE','KSS','CHCO','JAMF','ZYME','CXM','PRKS','UFPT','HLIO','ATRC','PAHC','NN','EXPI','VSH','ABR','ANDE','CMPR','NWBI','GTY','NNE','VVX','CCS','DFH','CLDX','IMNM','MCW','FDP','LILA','AMRC','NVTS','LILAK','CDRE','FLYW','CLMT','STOK','WBTN','FSLY','CC','WMK','HTZ','LZ','FCF','AMLX','BLBD','INVA','UTI','HTO','SEM','NEXT','WS','TWST','HLMN','PRCT','ENOV','PCT','PGEN','OMCL','BLX','REAL','NTST','STEL','CLBK','CCB','WT','BHE','MNKD','PWP','CASH','XPRO','INVX','LASR','PBI','AAMI','TILE','ELME','FLOC','SRCE','DGII','TCBK','HROW','EFC','NRIX','LKFN','SNDX','VITL','PGRE','EVCM','STBA','RHLD','AMPX','FLNG','AAOI','WINA','CNXN','MBIN','FIVN','TRVI','GABC','NSSC','HOUS','COLL','ICFI','ARLO','VRE','ROG','PAR','FUN','AZTA','IMKTA','UTZ','ORKA','LOB','VYX','SBH','EVEX','OLMA','UPB','ARI','NBHC','ARDX','WSR','NEO','NESR','DXPE','NABL','ADEA','KMTS','DCO','LADR','SPB','ELVN','TRS','ROCK','RAPP','VMEO','STGW','KW','SILA','WKC','GBX','DVAX','XHR','QCRH','AMPL','PZZA','HOPE','ARHS','UMH','CNMD','ENR','TNC','OUST','JCAP','CTOS','ARDT','PRA','IDT','SVV','DSGR','NVGS','AHCO','VRNT','PEB','MBX','BBNX','BY','MBC','HSII','PLAB','PSIX','CTS','WWW','HCSG','PFBC','TFIN','INBX','AIN','THS','DFIN','NSP','URGN','GCT','AXGN','FUBO','BFC','CNOB','BLFS','XERS','MXL','XPEL','WTTR','ESRT','FOR','MVST','EYPT','WABC','ZD','WVE','PHR','DCOM','DEC','AAT','ACVA','ALGT','ALEX','GRC','RBCAA','RES','GIII','NAVI','CRAI','ORIC','XNCR','BV','PVLA','TNGX','SAFT','PRLB','CSR','RLJ','METC','SBGI','PRO','RLAY','TSHA','DHC','OMDA','THR','IRMD','AESI','ORC','ALX','ASTH','NVRI','SFL','MNMD','VOYG','LPG','PMT','VTOL','ECPG','PHAT','ALIT','REX','OBK','SLDP','JBGS','SEPN','PCRX','VRTS','CRI','GIC','PRG','THRM','EVLV','ANAB','USPH','COHU','CSTL','PEBO','PDM','PX','TTI','OCFC','OPK','LGIH','PLYM','PUMP','TWO','CBL','NMAX','CIM','NRDS','BHVN','NPKI','CWH','CRCT','DEA','RZLV','PDFS','CTBI','SCL','PRCH','MLKN','FWRG','PRSU','CRML','ECVT','PENG','SVRA','UAMY','OSBC','MDXG','RZLT','PLPC','SAFE','HLF','HLX','UVE','TMP','ASTE','GOGO','UFCS','GLUE','UPBD','BH','HNRG','FTRE','WGO','PNTG','RDW','BHRB','TBPH','MRVI','FLGT','BBSI','MSEX','IOVA','RPD','PLSE','LENZ','MEG','BORR','CMPX','ASIC','TYRA','TDUP','HRTG','TK','BOW','LDI','IART','DAWN','EIG','FSUN','FMBH','DAKT','ULCC','ALNT','UTL','AEBI','UVSP','EVER','SBSI','BCAX','SCSC','CCOI','WLFC','DLX','ODP','SENEA','ESQ','AVBP','AMAL','EMBC','CDNA','OPFI','ABUS','JBI','AIV','VTS','CFFN','FBRT','GLDD','ATLC','ACEL','SMP','AVO','HBNC','JBSS','SIBN','ERAS','LVWR','IIIV','NAT','LMB','RCAT','CRMD','EQBK','CTLP','BLND','NXRT','HAFC','WOOF','CTEV','NBBK','MOFG','CPF','MRTN','GDEN','INR','MATW','RIGL','GNK','EEX','GSM','AMSF','INDI','HFWA','MAX','CNNE','REAX','BETR','QNST','PSNL','BFS','RDVT','AXL','ERII','SWIM','ODC','BFST','BJRI','TRST','GOSS','SKYT','HSTM','APOG','VSTS','NB','GYRE','HBT','SSTK','NGVC','MCB','VEL','MBWM','LQDT','ASPI','ATXS','FSBC','AMTB','REPL','CCNE','AQST','AMWD','HOV','KALV','BTBT','SCHL','BRSP','VTEX','DGICA','NTGR','GERN','CLB','GPRE','INNV','GDYN','SPRY','HPP','HTB','TIPT','TRTX','ALMS','WRLD','KODK','PLOW','TREE','LINC','CTKB','NBR','PRME','SERV','NPK','LXEO','CCBG','KE','CRVS','TH','ESPR','SOC','SKYH','RYI','VTLE','MTUS','THFF','NBN','MATV','CARS','ORGO','GRNT','ACDC','MPB','LIND','GRPN','CSV','CAC','NUTX','RWT','GSBC','KOS','FULC','LXU','MYE','MCBS','RR','APEI','NXDR','PRAA','VSTM','HTBK','ADAM','AMRK','BBW','INN','IBCP','FDMT','TE','SRDX','GHM','MGTX','MYGN','BFLY','FRGE','SNDA','CBRL','SNCY','SPFI','JBIO','SMBC','OMER','BDN','ADTN','GETY','AIOT','IIIN','BCAL','SMBK','DSP','GDOT','CRSR','BZH','ANNX','AEVA','ABL','OFIX','FCBC','BWMN','FISI','AIP','ETD','NPWR','REPX','FOXF','MNPR','CTO','ALTI','HIFS','RGNX','BVS','ACIC','UHT','NAGE','HTLD','KOP','CMCL','NPB','KRUS','MNRO','SHEN','CWCO','PRTA','IVR','SFIX','ZVRA','TCMD','SXC','BKKT','LYTS','IHRT','ALRS','GOOD','KREF','IDR','CPS','SHBI','BKSY','NWPX','ASC','CRD.A','SD','ABAT','BLMN','CASS','AVNS','KROS','EBS','IRWD','NX','EGBN','IBTA','GTN','ICHR','PKST','FFIC','KFRC','TALK','TCBX','RPAY','WASH','NRIM','CDZI','KRO','CBLL','ITIC','CGEM','APPS','RBBN','ADCT','PSTL','TRNS','AHH','LAB','HZO','MGPI','ANGO','EU','GBFH','ALT','RSVR','HY','SB','FMNB','HCKT','AROW','BHB','UNTY','ACNB','RMR','CLMB','OXM','MBUU','PFIS','NPCE','PLAY','TROX','IBEX','DC','CLNE','GEVO','RGR','DOMO','OSPN','NUS','FIP','SCVL','CVLG','ARKO','CEVA','TWI','DRUG','BYND','HLLY','LEGH','ANGI','YORW','RRBI','BWB','MCS','ETON','IMXI','VLGEA','PGC','WEAV','WOW','EOLS','BNTC','LE','OLP','SMC','CBNK','FRPH','NLOP','GLRE','FTK','CHCT','ABSI','COFS','CODI','VREX','KLC','EBF','AMBQ','AKBA','BXC','TRC','NFBK','KRT','KOPN','CYRX','PANL','SIGA','STKL','BAND','DSGN','PRTH','MLR','TLS','CLFD','KIDS','FBIZ','AII','KRNY','XOMA','HBCP','FFWM','MAMA','CIVB','CWBC','FPI','CMCO','FVR','VPG','NEXN','CYH','EHAB','CMTG','CCSI','WEST','BMRC','CLPT','LAW','BBBY','RC','AMBC','MITK','EVH','BYRN','SFST','TCI','RXST','ZEUS','ATEX','MBI','MLAB','NGS','BSRR','JOUT','ASIX','NATH','BOC','PSFE','BSVN','SSP','DIN','HSHP','DMAC','GNE','CCRN','ZUMZ','EGY','FRBA','CARE','GCBC','OPAL','RYAM','PKE','MTW','NCMI','BRBS','MG','SRTA','NRC','HVT','OIS','SGHT','DH','ORN','SWBI','SITC','WTBA','PDLB','KFS','CMRC','ILPT','TTGT','ZIP','ULH','AMCX','BGS','TITN','NC','ONIT','ENTA','TECX','LAND','SLDB','CRNC','RM','NATR','BWFG','USNA','RMNI','DHIL','CZNC','AURA','CMDB','VHI','VALU','ACTG','FET','LWAY','ADV','TRDA','OBT','CVGW','CTRN','PBFS','NGNE','MSBI','IPI','CAL','SLP','TOI','GCO','MVBF','LZM','CTGO','SBC','RBB','FMAO','DENN','RCKT','CLDT','USCB','CARL','MTRX','AVXL','LXFR','MVIS','TBRG','MEC','WNC','LOCO','BBCP','BCML','VNDA','PCB','BRCC','DCTH','SMLR','ACCO','ATNI','OOMA','KELYA','HYLN','SI','LRMR','FSBW','BIOA','ALDX','PLBC','CIA','CERS','RNGR','MOV','PAYS','HDSN','EGAN','COSO','VOXR','MCFT','XPOF','NVEC','ASLE','TTSH','MPX','AVNW','CBAN','CNDT','WSBF','LFCR','QTRX','NEWT','WTI','MLP','TG','ALLO','RLGT','MAGN','WALD','FSTR','CIO','FNLC','SNWV','SPIR','FEIM','NODK','FHTX','CIX','PKOH','QSI','ELA','NECB','CLW','STRT','BRT','JMSB','LNKB','SEG','AIRO','ACRE','SVC','CURI','CAPR','OPRX','QUAD','MBCN','JACK','FRST','ACRS','PCYO','XPER','OFLX','NL','TBCH','WEYS','FOA','BRY','CZFS','REFI','RXT','JRVR','OEC','SPOK','TSBK','BLZE','EVI','MKTW','BZAI','TRAK','TARA','ASPN','AFRI','ARL','FDBC','NWFL','AIRS','EB','GWRS','PBYI','MPAA','SLQT','PKBK','III','AVIR','CHMG','FINW','MITT','CADL','NREF','RMAX','THRY','AIRJ','ALCO','MEI','EVC','FBLA','CVRX','TCX','WNEB','BKTI','PNRG','VMD','TKNO','MNTK','MDWD','FENC','PINE','OABI','LMNR','OVLY','VYGR','FUNC','KLTR','ALMU','ISTR','BNED','TLSI','AARD','PDYN','ABEO','MYFW','USAU','PESI','ONTF','ASUR','FVCB','CFFI','MFIN','TSSI','AGL','HWBK','FRAF','RGCO','BPRN','OMI','RCKY','HUMA','SLS','LCNB','DOUG','JILL','FCCO','ELMD','FXNC','STXS','VABK','DERM','FLWS','AOMR','OPRT','SNFCA','FSFG','EWCZ','MASS','KRMD','SMHI','PAL','POWW','TNXP','RICK','SWKH','CBNA','PAMT','SEVN','SKYX','TRUE','FLXS','ZVIA','HBB','LFMD','IMMR','STRZ','OPBK','RNAC','EFSI','GENC','ATLO','INSE','CXDO','ONEW','INGN','FC','NKSH','MXCT','BHR','SATL','UIS','CPSS','NXXT','VUZI','TMCI','AREN','FTLF','SMTI','NRDY','LPRO','ESCA','UTMD','CATX','SMID','ARCT','LOVE','DBI','JAKK','GAMB','UBFO','OVBC','MRBK','CBFV','SAMG','SKIN','PEBK','FNKO','SLND','BLFY','INSG','HNVR','FNWD','CZWI','STRW','RGP','OSUR','LARK','INBK','ACR','NVCT','DMRC','CMT','FCAP','ECBK','JELD','NXDT','ESOA','STRS','MDV','FFAI','CRMT','BVFL','WHG','HCAT','ARQ','CDXS','CFBK','AEYE','PLX','ALTG','FRD','KGEI','CHRS','LUCD','AVR','PNBK','MNSB','RELL','RCMT','BFIN','SGC','ALEC','FF','RMBI','EXFY','ANIK','ACU','TBI','ACTU','PMTS','SLSN','EPM','MPTI','CRDF','LAKE','SVCO','BCBP','ACNT','VGAS','FORR','EBMT','TTEC','NKTX','AVD','BSET','SBFG','HQI','SAFX','CLAR','BEEP','SUNS','JYNT','MAPS','RCEL','ISPR','LNSR','BARK','SEAT','GMGI','MED','HFFG','SIEB','SRBK','RPT','IKT','EML','SFBC','BOOM','GLSI','EHTH','BTMD','RBKB','AISP','CURV','CSPI','COOK','KULR','KG','EP','RVSB','HAIN','ARAY','EPSN','STIM','PROP','VIRC','GAIA','FSP','UNB','PDEX','HURA','ELDN','DCGO','SKIL','TUSK','OM','TVGN','AOUT','ILLR','LFVN','MYPS','SSTI','LFT','TZOO','ATOM','TEAD','ATYR','GOCO','AFCG','LAZR','CLPR','KRRO','LUNG','SNCR','INMB','TVRD','NEON','MYO','TSE','ZSPC','A','AA','AAL','AAON','AAP','AAPL','AAXJ','AB','ABBV','ABC','ABCL','ABEV','ABNB','ABT','ACGL','ACHR','ACI','ACLS','ACM','ACN','ADBE','ADC','ADI','ADM','ADNT','ADP','ADSK','AEE','AEHR','AEM','AEP','AER','AES','AFL','AFRM','AG','AGCO','AGG','AIG','AIMC','AIR','AIZ','AJG','AKRO','ALB','ALGM','ALGN','ALK','ALL','ALLE','ALLY','ALNY','ALSN','ALV','AMAT','AMBA','AMC','AMCR','AMD','AME','AMED','AMG','AMGN','AMH','AMKR','AMN','AMP','AMPH','AMSC','AMT','AMT?','AMX','AMZN','ANET','ANSS','AON','AOS','AOSL','APA','APD','APH','APO','APP','APPF','APPN','APTV','AR','ARCB','ARCC','ARCE','ARE','ARES','ARGX','ARMK','ARR?','ARRY','ARVN','ASAN','ASGN','ASHR','ASM','ASML','ASND','ASO','ASR','ASTR','ASTS','ASX','ASYS','ATEN','ATI','ATUS','ATVI','AU','AVAL','AVB','AVGO','AVNT','AVTR','AWK','AWR','AXNX','AXON','AXP','AXTA','AXTI','AYX','AZN','AZO','BA','BABA','BAC','BALL','BALY','BAR','BAX','BBAR','BBBYQ','BBD','BBIG','BBVA','BBWI','BBY','BCRX','BDX','BE','BEAM','BECT?','BEKE','BEN','BERY','BF.B','BG','BHC','BHP','BIDU','BIG','BIIB','BILI','BILL','BIO','BIRK','BITO','BK','BKCH','BKNG','BKR','BKU','BKV','BLD','BLDP','BLDR','BLK','BLNK','BLOK','BMA','BMBL','BMO','BMRN','BMY','BND','BNS','BNTX','BOH','BOX','BP','BR','BRBR','BRKR','BRKS','BRO','BRX','BSBR','BSX','BTG','BTI','BUD','BURL','BVN','BWA','BWXT','BX','BXMT','BXP','C','CAAP','CABO','CACC','CAG','CAH','CAMT','CAR','CARR','CART','CASY','CAT','CATY','CAVA','CB','CBOE','CBRE','CBSH','CCEP','CCI','CCJ','CCL','CDAY','CDE','CDNS','CDW','CE','CEG','CEIX','CELH','CEPU','CF','CFG','CFLT','CFR','CFX','CG','CHD','CHDN','CHE','CHK','CHKP','CHPT','CHRD','CHRW','CHTR','CHWY','CI','CIB','CIEN','CINF','CIVI','CL','CLF','CLH','CLOV','CLVT','CLX','CMA','CMC','CMCSA','CME','CMG','CMI','CMP','CNA','CNC','CNI','CNP','CNQ','CNX','COF','COHR','COIN','COKE','COLM','COO','COP','CORT','COST','COTY','COUP','COUR','CP','CPE','CPNG','CPP','CPRI','CPRT','CPRX','CPT','CRDO','CRH','CRL','CRM','CROX','CRS','CRSP','CRUS','CRWD','CS','CS?','CSCO','CSGS','CSL','CSX','CTAS','CTLT','CTRA','CTSH','CTVA','CUBE','CUM','CVNA','CVS','CVX','CW','CWEN','CWT','CXW','CYBR','CZR','CZZ','D','DAL','DAN','DAO','DASH','DAY','DBA','DBB','DBC','DBX','DCBO','DD','DDD','DDOG','DE','DECK','DEI','DELL','DEO','DESP','DFS','DG','DGX','DHI','DHR','DIA','DINO','DIOD','DIS','DISH','DJCO','DKNG','DKS','DLB','DLO','DLR','DLTR','DM','DNA','DNN','DNUT','DOC','DOCN','DOCS','DOCU','DOLE','DORM','DOW','DOX','DOYU','DPZ','DRI','DSKE','DT','DTM','DUK','DUOL','DV','DVA','DVN','DXC','DXCM','EA','EADSY?','EAT','EBAY','EC','ECL','ED','EDIT','EDR','EDU','EDV','EEM','EEMS','EEMV','EFA','EG','EGHT','EHC','EIX','EL','ELAN','ELF','ELS','ELV','EME','EMN','EMR','EMXC','ENB','ENPH','ENS','ENTG','ENV','ENVX','EOG','EPAM','EPC','EPD','EPRT','EQH','EQIX','EQNR','EQR','EQT','ERIC','ERIE','ERJ','ERO','ES','ESNT','ESS','ESTC','ET','ETN','ETR','ETSY','EVGo','EVR','EVRG','EW','EWA','EWC','EWD','EWH','EWI','EWJ','EWK','EWL','EWN','EWP','EWQ','EWS','EWT','EWU','EWY','EWZ','EXAS','EXC','EXLS','EXP','EXPD','EXPE','EXR','F','FANG','FAST','FATE','FCX','FDS','FDX','FE','FERG','FEZ','FHN','FI','FICO','FIGS','FIS','FISV','FITB','FIVE','FIX','FLNC','FLR','FLT','FMC','FN','FND','FNF','FNV','FOUR','FOX','FOXA','FR','FREY','FRT','FSLR','FSR','FTNT','FTV','FWRD','FXI','GARMIN','GATX','GBTC','GD','GDDY','GDS','GE','GEF','GEHC','GEL','GEN','GEV','GEV?','GFL','GFS','GGAL','GGB','GGG','GH','GIL','GILD','GIS','GLBE','GLD','GLOB','GLPI','GLT?','GLW','GM','GMAB','GME','GMED','GMRE','GNRC','GNTX','GO','GOEV','GOL','GOLD','GOOG','GOOGL','GPC','GPI','GPK','GPN','GPRK','GRAB','GRMN','GRWG','GS','GSAT','GSK','GT','GTES','GTLB','GTLS','GVA','GWRE','GWW','GXO','H','HAL','HALO','HARLEY','HAS','HBAN','HBM','HCA','HCI','HCP','HD','HDB','HEI','HEI.A','HEINY','HELE','HES','HIBB','HIG','HII','Hikma','Hikma?','HIMS','HIPO','HL','HLIT','HLN','HLT','HMC','HMY','HNST','HOG','HOLX','HON','HOOD','HPE','HPK','HPQ','HR','HRL','HRTX','HSBC','HSIC','HST','HSY','HTHT','HUBB','HUBG','HUBS','HUM','HUT','HUYA','HWM','HXL','HYG','IAC','IAS','IAU','IAUX','IBIT','IBKR','IBM','IBN','ICE','ICLR','ICPT','IDXX','IEF','IEFA','IEUR','IEX','IFF','IGT','IIPR','IIVI','IJH','IJJ','IJK','IJR','IJS','IJT','ILMN','IMCR','INCY','INFA','INFY','ING','INMD','INSP','INTA','INTC','INTU','INVH','IONQ','IONS','IOT','IP','IPAR','IPGP','IQ','IQV','IR','IRDM','IRM','ISRG','IT','ITCI','ITGR','ITRI','ITT','ITUB','ITW','IVE','IVOL','IVW','IWM','IWN','IWO','J','JAZZ','JBHT','JBL','JBLU','JBT','JCI','JD','JKHY','JLL','JNJ','JNK','JNPR','JOBY','JPM','JWN','K','KALU','KBR','KD','KDP','KEN','KEP','KEY','KGC','KHC','KIM','KINS','KKR','KLAC','KMB','KMI','KMPR','KMPR?','KMX','KNX','KO','KOD','KOSS','KR','KRG','KRNT','KRTX','KRYS','KSU','KTB','KTOS','KURA','KVYO','KWEB','KXIN','L','L3Harris','LAC','LAD','LAMR','LANC','LBRDA','LC','LCID','LDOS','LEA','LEG','LEGN','LEN','LEU','LEV','LEVI','LFUS','LH','LHX','LI','LII','LIN','LITE','LKQ','LLY','LMND','LMT','LNC','LNG','LNN','LNT','LOGI','LOPE','LOW','LPLA','LQD','LRCX','LSCC','LSTR','LTBR','LTH','LTHM','LULU','LUMN','LUNR','LUV','LVS','LW','LYB','LYFT','LYV','LZB','MA','MAA','MAIN','MANH','MANU','MAR','MARA','MAS','MASI','MAT','MATX','MAXN','MBLY','MCD','MCHI','MCHP','MCK','MCO','MDB','MDGL','MDLZ','MDT','MDY','MDYG','MDYV','MELI','MET','META','MFA','MGA','MGM','MGNI','MHK','MKC','MKTX','MLI','MLM','MMC','MMI','MMM','MNDY','MNST','MO','MOH','MORN','MOS','MPC','MPLX','MPW','MPWR','MQ','MRK','MRNA','MRO','MRVL','MS','MSA','MSCI','MSFT','MSGE','MSGS','MSI','MSM','MSTR','MT','MTB','MTCH','MTD','MTDR','MTG','MTN','MTSI','MTUM','MTZ','MU','MULN','MUR','NAV?','NCLH','NCNO','NCR','NDAQ','NDSN','NEE','NEM','NEOG','NET','NEU','NFE','NFLX','NG','NGD','NIO','NKE','NKLA','NLY','NMIH','NNDM','NNOX','NOBL','NOC','NOK','NOMD','NOV','NOVA','NOW','NRG','NSC','NTAP','NTCT','NTES','NTLA','NTR','NTRA','NTRS','NU','NUE','NVAX','NVCR','NVDA','NVEI','NVMI','NVO','NVR','NVTA?','NWL','NWS','NWSA','NXPI','NXT','NYCB','NYT','O','ODFL','OGN','OHI','OKE','OKTA','OLLI','OLN','OLPX','OMC','ON','ONB','ONON','ONTO','OPCH','ORCL','ORI','ORLY','ORRF','OSCR','OSK','OTIS','OWL','OXY','PAA','PAAS','PAC','PACB','PACK','PACW','PAGS','PALL','PANW','PARA','PARR','PATH','PAYC','PAYO','PAYX','PBH','PBR','PBR.A','PCAR','PCG','PCOR','PCTY','PD','PDCO','PDD','PEAK','PEG','PEN','PENN','PEP','PERI','PFE','PFG','PG','PGR','PH','PHG','PHM','PINS','PJT','PKG','PKX','PL','PLD','PLL','PLNT','PLTK','PLTR','PLUG','PM','PNC','PNFP','PNR','PNW','POOL','POR','POST','POWI','PPG','PPL','PPLT','PR','PRAH?','PRAX','PRGO','PRGS','PRU','PSA','PSN','PSQ','PSX','PTC','PTEN','PTLO','PTON','PUBM','PVH','PWR','PWSC','PXD','PYPD','PYPL','QCOM','QDEL','QID','QLD','QQQ','QRVO','QS','QSR','QTWO','QUAL','QUOT','R','RACE','RAD','RBLX','RCL','RDDT','RDN','RE','REG','REGN','RELX','RF','RGA','RGEN','RGTI','RH','RHI','RILY','RIO','RIOT','RIVN','RJF','RKLB','RKT','RL','RMBS','RMD','RNG','RNR','ROK','ROKU','ROL','ROOT','ROP','ROST','RPRX','RRC','RRX','RS','RSG','RSP','RTX','RUN','RVLV','RVTY','RXRX','RY','RYAAY','S','SA','SABR','SAGE','SAH','SAIA','SAM','SAN','SANA','SAP','SATS','SAVA','SAVE','SBS','SBSW','SBUX','SCCO','SCHA','SCHD','SCHE','SCHF','SCHG','SCHM','SCHW','SDA','SDGR','SDS','SE','SEAS','SEDG','SEE','SEIC','SF','SFM','SG','SGEN','SGH','SGML','SGMO','SH','SHAK','SHEL','SHLS','SHOO','SHOP','SHW','SHY','SID','SIG','SIMO','SIRI','SITM','SIVB?','SIVBQ','SIVR','SJM','SJW','SKX','SLAB','SLB','SLG','SLGN','SLV','SLY','SLYG','SLYV','SM','SMAR','SMCI','SMG','SMR','SMTC','SNA','SNAP','SNBR','SNDR','SNOW','SNPS','SNV','SNX','SNY','SO','SOFI','SON','SONO','SONY','SPCE','SPG','SPGI','SPLG','SPLK','SPOT','SPR','SPSC','SPT','SPWR','SPXU','SPY','SQ','SQM','SQQQ','SRE','SRPT','SSNC','SSO','SSYS','STAA','STAG','STE','STEM','STLA','STLD','STNE','STT','STX','STZ','SU','SUI','SUM','SUN','SUNW','SUPN','SUPV','SWAV','SWK','SWKS','SWN','SYF','SYK','SYY','T','TAK','TAL','TAP','TCEHY','TCOM','TD','TDG','TDOC','TDY','TEAM','TECH','TECK','TEF','TEL','TENB','TER','TEVA','TFC','TFII','TFX','TGNA','TGS','TGT','THC','TIP','TJX','TKR','TLRY','TLT','TM','TME','TMHC','TMO','TMUS','TNA','TNDM','TNET','TOL','TOST','TOT','TOY?','TPC','TPIC','TPR','TPX','TQQQ','TREX','TRGP','TRIP','TRMB','TRNO','TROW','TRP','TRTN','TRU','TRUP','TRV','TSCO','TSLA','TSM','TSN','TT','TTC','TTD','TTEK','TTWO','TUP','TUYA','TV','TWLO','TX','TXG','TXN','TXRH','TXT','TYL','TZA','U','UAA','UAL','UBER','UBS','UBSI','UCTT','UDMY','UDR','UE','UEC','UGA','UGP','UHS','UI','UL','ULTA','UMBF','UMC','UNF','UNG','UNH','UNIT','UNM','UNP','UPRO','UPS','UPST','URBN','URI','UROY','USB','USMV','USO','UUUU','UVV','V','VAC','VAL','VALE','VB','VBK','VBR','VC','VCEL','VCTR','VEA','VECO','VEEV','VERV','VFC','VGK','VICI','VIG','VIPS','VIR','VIST','VKTX','VLO','VMC','VMI','VNO','VNT','VO','VOO','VOYA','VPL','VRM','VRNS','VRSK','VRSN','VRT','VRTX','VSAT','VSCO','VST','VT','VTI','VTIP','VTR','VTRS','VTV','VTWO','VUG','VV','VYM','VZ','W','WAB','WAL','WAT','WBA','WBD','WBS','WCN','WDAY','WDAY?','WDC','WEC','WELL','WEN','WERN','WEX','WFC','WING','WIT','WIX','WLDN','WLK','WM','WMB','WMS','WMT','WOLF','WORK','WPC','WPM','WRB','WRBY','WRK','WSC','WTRG','WTW','WWD','WWE','WYN?','WYNN','X','XEL','XLB','XLC','XLE','XLF','XLI','XLK','XLNX?','XLP','XLRE','XLU','XLV','XLY','XOM','XPEV','XPO','XRAY','XRX','XXII','XYL','YELP','YETI','YEXT','YJ','YPF','YUM','YUMC','YY','Z','ZBH','ZBRA','ZI','ZIM','ZION','ZIOP?','ZKH?','ZLAB','ZM','ZNTL','ZROZ','ZS','ZSAN','ZTO','ZTS','ZUO',


]

PERIODO_ANOS_ENTRENAMIENTO = 3          # años de historia para cada loop
FWD_DAYS = 20                           # días hacia adelante p/target de entrenamiento (clasificador)
FWD_DAYS_TARGET = 7                     # días hacia adelante p/retorno “una semana”
N_SEMANAS_BACKTEST = 1                 # ← por ahora 5; si exporta ok, subimos a 100
OUTPUT_XLSX = Path("Consolidado_1_semanas_todos los tickers_1.xlsx")
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
