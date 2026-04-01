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
   
    
   'SPY', 'KBDC', 'KBON', 'KBONU', 'KBONW', 'KBSX', 'KBWP', 'KBWY', 'KC', 'KCHV', 'KCHVR', 'KCHVU', 'KDK', 'KDKRW', 'KELYB', 'KEQU', 'KEX', 'KEYS', 'KF', 'KFFB', 'KFII', 'KFIIR', 'KFIIU', 'KIDZ', 'KIDZW', 'KIO', 'KITT', 'KITTW', 'KKRS', 'KKRT', 'KLAR', 'KLRS', 'KLTO', 'KLTOW', 'KLXE', 'KMDA', 'KMPB', 'KMRK', 'KNDI', 'KNO', 'KNOP', 'KNRX', 'KNSA', 'KNSL', 'KOF', 'KONG', 'KOOL', 'KORE', 'KOYN', 'KOYNU', 'KOYNW', 'KPLT', 'KPLTW', 'KPRX', 'KPTI', 'KRAQU', 'KRC', 'KRKR', 'KRMN', 'KRP', 'KRSP', 'KSCP', 'KSPI', 'KT', 'KTCC', 'KTEC', 'KTF', 'KTH', 'KTN', 'KTTA', 'KTTAW', 'KTWOU', 'KUST', 'KVAC', 'KVACW', 'KVHI', 'KVUE', 'KWM', 'KWMWW', 'KYIV', 'KYIVW', 'KYN', 'KYNB', 'KYTX', 'KZIA', 'KZR', 'LAES', 'LAFA', 'LAFAR', 'LAFAU', 'LANDO', 'LANDP', 'LANV', 'LAR', 'LASE', 'LATA', 'LATAU', 'LATAW', 'LAZ', 'LB', 'LBGJ', 'LBRDK', 'LBRDP', 'LBRX', 'LBTYA', 'LBTYB', 'LBTYK', 'LCAP', 'LCCC', 'LCCCR', 'LCCCU', 'LCFY', 'LCFYW', 'LCTX', 'LCUT', 'LDEM', 'LDP', 'LDRH', 'LECO', 'LEDS', 'LEE', 'LEGT', 'LEN.B', 'LENS', 'LEO', 'LESL', 'LEXX', 'LFAC', 'LFACU', 'LFACW', 'LFMDP', 'LFS', 'LFWD', 'LGCB', 'LGCL', 'LGCY', 'LGHL', 'LGI', 'LGL', 'LGN', 'LGO', 'LGPS', 'LGVN', 'LHAI', 'LHSW', 'LICN', 'LIDR', 'LIDRW', 'LIEN', 'LIFE', 'LIMN', 'LIMNW', 'LINE', 'LINK', 'LIQT', 'LITB', 'LITM', 'LITS', 'LIVE', 'LIXT', 'LKSP', 'LKSPR', 'LKSPU', 'LLYVA', 'LLYVK', 'LMFA', 'LMNX', 'LMRI', 'LNAI', 'LND', 'LNKS', 'LNZA', 'LNZAW', 'LOAN', 'LOAR', 'LOBO', 'LOCL', 'LODE', 'LOKV', 'LOKVU', 'LOKVW', 'LOMA', 'LONA', 'LOOP', 'LOT', 'LOTWW', 'LPA', 'LPAA', 'LPAAU', 'LPAAW', 'LPBB', 'LPBBU', 'LPBBW', 'LPCN', 'LPCV', 'LPCVU', 'LPCVW', 'LPL', 'LPSN', 'LPTH', 'LPX', 'LRE', 'LRHC', 'LSAK', 'LSBK', 'LSE', 'LSF', 'LSH', 'LSPD', 'LSTA', 'LTCC', 'LTM', 'LTRN', 'LTRX', 'LTRYW', 'LU', 'LUCK', 'LUCY', 'LUCYW', 'LUD', 'LUXE', 'LVLU', 'LVO', 'LVRO', 'LVROW', 'LWAC', 'LWACU', 'LWACW', 'LWLG', 'LX', 'LXEH', 'LXRX', 'LYEL', 'LYG', 'LYRA', 'LZMH', 'M', 'MAAS', 'MACI', 'MACIU', 'MACIW', 'MAGS', 'MAIA', 'MAMO', 'MAN', 'MANE', 'MAPP', 'MAPSW', 'MARPS', 'MASK', 'MATH', 'MAYS', 'MB', 'MBAI', 'MBAV', 'MBAVU', 'MBAVW', 'MBB', 'MBBC', 'MBINL', 'MBINM', 'MBINN', 'MBIO', 'MBND', 'MBNKO', 'MBOT', 'MBRX', 'MBVI', 'MBVIU', 'MBVIW', 'MCGA', 'MCGAU', 'MCGAW', 'MCHPP', 'MCHX', 'MCI', 'MCN', 'MCR', 'MCRB', 'MCRP', 'MDAI', 'MDAIW', 'MDBH', 'MDCX', 'MDCXW', 'MDIA', 'MDLN', 'MDRR', 'MDU', 'MDXH', 'MEDP', 'MEDX', 'MEGI', 'MEGL', 'MEHA', 'MEMS', 'MENS', 'MEOH', 'MERC', 'MESH', 'MESHU', 'MESHW', 'MESO', 'METCB', 'METCI', 'METCZ', 'MEVOU', 'MFAN', 'MFAO', 'MFC', 'MFG', 'MFI', 'MFIC', 'MFICL', 'MFLX', 'MFM', 'MGF', 'MGIC', 'MGIH', 'MGLD', 'MGN', 'MGNX', 'MGR', 'MGRB', 'MGRD', 'MGRE', 'MGRT', 'MGRX', 'MGX', 'MGYR', 'MHD', 'MHF', 'MHH', 'MHLA', 'MHN', 'MHNC', 'MHY', 'MI', 'MIAX', 'MICC', 'MIDD', 'MIG', 'MIGI', 'MIMI', 'MIN', 'MIND', 'MINE', 'MIRA', 'MIST', 'MITN', 'MITP', 'MITQ', 'MIY', 'MKC.V', 'MKDW', 'MKDWW', 'MKL', 'MKLY', 'MKLYR', 'MKLYU', 'MKSI', 'MKZR', 'MLAAU', 'MLAC', 'MLACR', 'MLACU', 'MLCI', 'MLCIL', 'MLCO', 'MLEC', 'MLECW', 'MLGO', 'MLSS', 'MLTX', 'MMA', 'MMD', 'MMLP', 'MMT', 'MMTX', 'MMTXU', 'MMTXW', 'MMU', 'MMYT', 'MNDO', 'MNDR', 'MNOV', 'MNR', 'MNSBP', 'MNSO', 'MNTN', 'MNTS', 'MNTSW', 'MNY', 'MNYWW', 'MOB', 'MOBBW', 'MOBX', 'MOBXW', 'MODD', 'MODL', 'MOG.B', 'MOGU', 'MOLN', 'MOMO', 'MOVE', 'MP', 'MPA', 'MPG', 'MPLT', 'MPT', 'MPU', 'MPV', 'MQT', 'MQY', 'MRAM', 'MRCC', 'MREO', 'MRKR', 'MRM', 'MRNO', 'MRNOW', 'MRP', 'MRSH', 'MRT', 'MSAI', 'MSAIW', 'MSB', 'MSBIP', 'MSC', 'MSD', 'MSDL', 'MSGM', 'MSGY', 'MSIF', 'MSLE', 'MSN', 'MSS', 'MSTX', 'MSW', 'MTA', 'MTC', 'MTEK', 'MTEKW', 'MTEN', 'MTEX', 'MTLS', 'MTNB', 'MTR', 'MTVA', 'MUA', 'MUC', 'MUE', 'MUFG', 'MUJ', 'MUSA', 'MUX', 'MUZEU', 'MVF', 'MVO', 'MVSTW', 'MVT', 'MWG', 'MWH', 'MWYN', 'MX', 'MXC', 'MXE', 'MXF', 'MYD', 'MYI', 'MYN', 'MYND', 'MYNZ', 'MYPSW', 'MYSE', 'MYSEW', 'MYSZ', 'NA', 'NAAS', 'NAC', 'NAD', 'NAII', 'NAK', 'NAKA', 'NAMI', 'NAMM', 'NAMMW', 'NAMS', 'NAMSW', 'NAN', 'NAUT', 'NAVN', 'NAZ', 'NBB', 'NBH', 'NBIS', 'NBIX', 'NBP', 'NBRGU', 'NBTX', 'NBXG', 'NBY', 'NCA', 'NCDL', 'NCEL', 'NCEW', 'NCI', 'NCIQ', 'NCL', 'NCNA', 'NCPL', 'NCPLW', 'NCRA', 'NCSM', 'NCT', 'NCTY', 'NCV', 'NCZ', 'NDLS', 'NDMO', 'NDRA', 'NEA', 'NEGG', 'NEN', 'NEOV', 'NEOVW', 'NEPH', 'NERV', 'NETL', 'NEUP', 'NEWP', 'NEWTG', 'NEWTH', 'NEWTI', 'NEWTO', 'NEWTP', 'NEXA', 'NEXM', 'NFG', 'NFGC', 'NFJ', 'NGEN', 'NGG', 'NGL', 'NHIC', 'NHICU', 'NHICW', 'NHPAP', 'NHPBP', 'NHS', 'NHTC', 'NI', 'NICE', 'NIE', 'NIM', 'NINE', 'NIOBW', 'NIPG', 'NIQ', 'NITE', 'NITO', 'NIU', 'NIVF', 'NIVFW', 'NIXX', 'NIXXW', 'NKLR', 'NKTR', 'NKX', 'NMAI', 'NMAR', 'NMB', 'NMBL', 'NMCO', 'NMFC', 'NMFCZ', 'NMG', 'NMI', 'NML', 'NMM', 'NMP', 'NMPAR', 'NMPAU', 'NMR', 'NMRA', 'NMS', 'NMT', 'NMTC', 'NMZ', 'NNAVW', 'NNBR', 'NNN', 'NNNN', 'NNVC', 'NNY', 'NOA', 'NOAH', 'NOEM', 'NOEMR', 'NOEMW', 'NOM', 'NOMA', 'NOTE', 'NOTV', 'NOVTU', 'NP', 'NPAC', 'NPACU', 'NPACW', 'NPCT', 'NPFD', 'NPT', 'NPV', 'NQP', 'NRGV', 'NRK', 'NRO', 'NRP', 'NRSN', 'NRSNW', 'NRT', 'NRUC', 'NRXP', 'NRXPW', 'NRXS', 'NSA', 'NSPR', 'NSRX', 'NSTS', 'NSYS', 'NTCL', 'NTHI', 'NTIC', 'NTIP', 'NTNX', 'NTRB', 'NTRBW', 'NTRP', 'NTRSO', 'NTSK', 'NTWK', 'NTWO', 'NTWOU', 'NTWOW', 'NTZ', 'NUAI', 'NUAIW', 'NUV', 'NUW', 'NUWE', 'NVA', 'NVAWW', 'NVDQ', 'NVG', 'NVNI', 'NVNIW', 'NVNO', 'NVS', 'NVST', 'NVT', 'NVVE', 'NVVEW', 'NVX', 'NWAX', 'NWG', 'NWGL', 'NWTG', 'NXE', 'NXG', 'NXGL', 'NXGLW', 'NXJ', 'NXL', 'NXP', 'NXPL', 'NXPLW', 'NXST', 'NXTC', 'NXTT', 'NYAX', 'NYC', 'NYM', 'NYXH', 'NZF', 'OABIW', 'OACC', 'OACCU', 'OACCW', 'OAKU', 'OAKUR', 'OAKUW', 'OBA', 'OBAI', 'OBAWU', 'OBAWW', 'OBDC', 'OBE', 'OBIO', 'OBTC', 'OC', 'OCC', 'OCCI', 'OCCIM', 'OCCIN', 'OCCIO', 'OCG', 'OCGN', 'OCS', 'OCSAW', 'OCSL', 'ODD', 'ODV', 'ODVWZ', 'ODYS', 'OESX', 'OFAL', 'OFRM', 'OFS', 'OFSSH', 'OFSSO', 'OGE', 'OGEN', 'OGI', 'OIA', 'OILT', 'OIMAU', 'OKUR', 'OKYO', 'OLB', 'OLED', 'OLOX', 'OMAB', 'OMEX', 'OMF', 'OMH', 'OMSE', 'ONBPO', 'ONBPP', 'ONC', 'ONCH', 'ONCHU', 'ONCHW', 'ONCO', 'ONCY', 'ONDS', 'ONEG', 'ONFO', 'ONFOW', 'ONL', 'ONMD', 'ONMDW', 'OPAD', 'OPEN', 'OPENL', 'OPENW', 'OPENZ', 'OPHC', 'OPLN', 'OPP', 'OPRA', 'OPTT', 'OPTU', 'OPTX', 'OPTXW', 'OPXS', 'OPY', 'OR', 'ORBS', 'ORGN', 'ORGNW', 'ORIO', 'ORIQ', 'ORIQU', 'ORIQW', 'ORIS', 'ORKT', 'ORLA', 'ORMP', 'OS', 'OSG', 'OSRH', 'OSRHW', 'OSS', 'OSTX', 'OTEX', 'OTF', 'OTGA', 'OTGAU', 'OTGAW', 'OTH', 'OTLK', 'OTLY', 'OUSTZ', 'OVID', 'OVV', 'OWLS', 'OWLT', 'OXBR', 'OXBRW', 'OXLC', 'OXLCG', 'OXLCI', 'OXLCL', 'OXLCN', 'OXLCO', 'OXLCP', 'OXLCZ', 'OXSQ', 'OXSQG', 'OXSQH', 'OYSE', 'OYSER', 'OYSEU', 'OZ', 'OZK', 'OZKAP', 'PAACU', 'PACH', 'PACHU', 'PACHW', 'PAG', 'PAGP', 'PAI', 'PAII', 'PALI', 'PAM', 'PAPL', 'PARK', 'PASG', 'PASW', 'PAVM', 'PAVS', 'PAXS', 'PAY', 'PB', 'PBA', 'PBHC', 'PBM', 'PBMWW', 'PBT', 'PCAP', 'PCAPU', 'PCAPW', 'PCF', 'PCI', 'PCLA', 'PCLN', 'PCM', 'PCN', 'PCQ', 'PCR', 'PCS', 'PCSA', 'PCSC', 'PCTTW', 'PDC', 'PDCC', 'PDI', 'PDO', 'PDPA', 'PDS', 'PDSB', 'PDT', 'PDX', 'PDYNW', 'PED', 'PEGA', 'PELI', 'PELIR', 'PELIU', 'PEO', 'PEPG', 'PERF', 'PETS', 'PETZ', 'PEW', 'PFAI', 'PFD', 'PFGC', 'PFH', 'PFL', 'PFLT', 'PFN', 'PFO', 'PFSA', 'PFX', 'PFXNZ', 'PGAC', 'PGACR', 'PGACU', 'PGP', 'PGYWW', 'PGZ', 'PHAR', 'PHGE', 'PHI', 'PHIO', 'PHK', 'PHOE', 'PHUN', 'PHVS', 'PICS', 'PID', 'PIII', 'PIIIW', 'PIM', 'PIO', 'PK', 'PLAG', 'PLBL', 'PLBY', 'PLCE', 'PLG', 'PLMK', 'PLMKU', 'PLMKW', 'PLRX', 'PLRZ', 'PLSM', 'PLUR', 'PLUT', 'PLYX', 'PMAX', 'PMCB', 'PMEC', 'PMI', 'PML', 'PMM', 'PMN', 'PMNT', 'PMO', 'PMTR', 'PMTRU', 'PMTRW', 'PMTU', 'PMTV', 'PMTW', 'PMVP', 'PN', 'PNI', 'PNNT', 'POAS', 'POCI', 'PODC', 'PODD', 'POET', 'POLA', 'POLE', 'POLEW', 'POM', 'PONY', 'POW', 'POWR', 'POWWP', 'PPBT', 'PPC', 'PPCB', 'PPHC', 'PPIH', 'PPSI', 'PPT', 'PRE', 'PRENW', 'PRFX', 'PRH', 'PRHI', 'PRHIZ', 'PRI', 'PRLD', 'PRMB', 'PROF', 'PROK', 'PROV', 'PRPL', 'PRPO', 'PRQR', 'PRS', 'PRSO', 'PRT', 'PRTC', 'PRTS', 'PRZO', 'PSBD', 'PSEC', 'PSF', 'PSHG', 'PSIG', 'PSKY', 'PSNY', 'PSNYW', 'PSO', 'PSQH', 'PSTG', 'PSTR', 'PSTV', 'PTA', 'PTF', 'PTH', 'PTHS', 'PTLE', 'PTN', 'PTORU', 'PTRN', 'PTY', 'PUK', 'PULM', 'PURR', 'PVL', 'PW', 'PWER', 'PWRD', 'PXED', 'PXI', 'PXLW', 'PXS', 'PYT', 'PYXS', 'PZG', 'Q', 'QCLS', 'QETA', 'QETAR', 'QFIN', 'QGEN', 'QH', 'QIPT', 'QLTI', 'QLTY', 'QMCO', 'QNCX', 'QNRX', 'QNTM', 'QQQJ', 'QQQM', 'QQQX', 'QRHC', 'QSEA', 'QSEAR', 'QSEAU', 'QSIAW', 'QTI', 'QTTB', 'QUIK', 'QUMS', 'QUMSR', 'QUMSU', 'QURE', 'QVCC', 'QVCD', 'QVCGA', 'QVCGP', 'QXO', 'RA', 'RAAQ', 'RAAQU', 'RAAQW', 'RAC', 'RADX', 'RAIL', 'RAIN', 'RAINW', 'RAL', 'RAND', 'RANG', 'RANGR', 'RANGU', 'RANI', 'RAPT', 'RARE', 'RAVE', 'RAY', 'RAYA', 'RBA', 'RBC', 'RBNE', 'RBOT', 'RBRK', 'RCB', 'RCC', 'RCD', 'RCG', 'RCI', 'RCKTW', 'RCON', 'RCS', 'RCT', 'RDAC', 'RDACR', 'RDACU', 'RDAG', 'RDAGU', 'RDAGW', 'RDCM', 'RDGT', 'RDHL', 'RDI', 'RDIB', 'RDNW', 'RDWR', 'RDY', 'RDZN', 'RDZNW', 'REBN', 'RECT', 'REE', 'REED', 'REFR', 'REGCO', 'REGCP', 'REI', 'REKR', 'RENT', 'RENX', 'RERE', 'RETO', 'REVB', 'REVBW', 'REXR', 'REYN', 'RFAI', 'RFAIR', 'RFAMU', 'RFI', 'RFIL', 'RFL', 'RFM', 'RFMZ', 'RGC', 'RGLD', 'RGNT', 'RGS', 'RGT', 'RGTIW', 'RIBB', 'RIBBR', 'RILYG', 'RILYK', 'RILYL', 'RILYN', 'RILYP', 'RILYT', 'RILYZ', 'RIME', 'RITM', 'RITR', 'RIV', 'RJET', 'RKDA', 'RLI', 'RLMD', 'RLTY', 'RLX', 'RLYB', 'RMCF', 'RMCO', 'RMCOW', 'RMI', 'RMM', 'RMMZ', 'RMSG', 'RMSGW', 'RMT', 'RMTI', 'RNAZ', 'RNGT', 'RNGTU', 'RNGTW', 'RNIN', 'RNP', 'RNTX', 'RNW', 'RNWWW', 'RNXT', 'ROIV', 'ROLR', 'ROMA', 'RPC', 'RPGL', 'RPID', 'RPM', 'RQI', 'RRGB', 'RSF', 'RSKD', 'RSSS', 'RSVRW', 'RTAC', 'RTACU', 'RTACW', 'RTO', 'RUBI', 'RUMBW', 'RVMD', 'RVMDW', 'RVP', 'RVPH', 'RVSN', 'RVSNW', 'RVT', 'RVYL', 'RWAY', 'RWAYI', 'RWAYL', 'RWAYZ', 'RWTN', 'RWTO', 'RWTP', 'RWTQ', 'RYAN', 'RYDE', 'RYET', 'RYM', 'RYN', 'RYOJ', 'RZB', 'RZC', 'RZLVW', 'SAAQU', 'SABA', 'SABS', 'SABSW', 'SAC', 'SACH', 'SAGT', 'SAIC', 'SAIH', 'SAIHW', 'SAIL', 'SAJ', 'SANG', 'SAR', 'SARO', 'SAT', 'SATA', 'SATLW', 'SAV', 'SAY', 'SAZ', 'SBAC', 'SBCWW', 'SBDS', 'SBET', 'SBEV', 'SBFM', 'SBFMW', 'SBI', 'SBLK', 'SBLX', 'SBR', 'SBXD', 'SBXE', 'SCAG', 'SCAGW', 'SCCD', 'SCCE', 'SCCF', 'SCCG', 'SCD', 'SCI', 'SCII', 'SCIIR', 'SCIIU', 'SCKT', 'SCLX', 'SCLXW', 'SCM', 'SCNI', 'SCNX', 'SCOR', 'SCPQ', 'SCPQU', 'SCPQW', 'SCWO', 'SCYX', 'SCZM', 'SDAWW', 'SDG', 'SDHC', 'SDHI', 'SDHIR', 'SDHIU', 'SDHY', 'SDOT', 'SDST', 'SDSTW', 'SEATW', 'SEB', 'SEED', 'SEER', 'SEGG', 'SELF', 'SELX', 'SEMG', 'SEMI', 'SENEB', 'SENS', 'SER', 'SERA', 'SES', 'SEV', 'SFB', 'SFD', 'SFHG', 'SFWL', 'SFY', 'SGA', 'SGI', 'SGLY', 'SGMT', 'SGN', 'SGP', 'SGRP', 'SGU', 'SHAZ', 'SHC', 'SHFS', 'SHFSW', 'SHG', 'SHIM', 'SHIP', 'SHLD', 'SHMD', 'SHMDW', 'SHPH', 'SIDU', 'SIF', 'SIFY', 'SIGIP', 'SII', 'SILC', 'SILO', 'SIM', 'SIMA', 'SIMAU', 'SIMAW', 'SINT', 'SITE', 'SJ', 'SJT', 'SKBL', 'SKE', 'SKK', 'SKLZ', 'SKM', 'SKYE', 'SKYQ', 'SLAI', 'SLDPW', 'SLE', 'SLF', 'SLGB', 'SLGL', 'SLI', 'SLM', 'SLMBP', 'SLMT', 'SLN', 'SLNG', 'SLNH', 'SLNHP', 'SLQD', 'SLRC', 'SLSR', 'SLVO', 'SLVR', 'SLXN', 'SLXNW', 'SMAP', 'SMFG', 'SMJF', 'SMMT', 'SMRT', 'SMSI', 'SMTK', 'SMU', 'SMWB', 'SMX', 'SMXT', 'SMXWW', 'SN', 'SNAL', 'SND', 'SNDK', 'SNDL', 'SNES', 'SNGX', 'SNN', 'SNOA', 'SNPX', 'SNSE', 'SNT', 'SNTG', 'SNTI', 'SNYR', 'SOAR', 'SOBO', 'SOBR', 'SOCA', 'SOCAU', 'SOCAW', 'SOGP', 'SOHOB', 'SOHON', 'SOHOO', 'SOHU', 'SOJC', 'SOJD', 'SOJE', 'SOJF', 'SOLC', 'SOLR', 'SOLS', 'SOLV', 'SOMN', 'SONM', 'SOPA', 'SOPH', 'SOR', 'SORA', 'SORNU', 'SOS', 'SOTK', 'SOUL', 'SOUNW', 'SOWG', 'SPAI', 'SPAQ', 'SPBC', 'SPCB', 'SPE', 'SPEG', 'SPEGR', 'SPEGU', 'SPH', 'SPHL', 'SPKL', 'SPKLW', 'SPLS', 'SPMA', 'SPMC', 'SPME', 'SPPL', 'SPRB', 'SPRC', 'SPRO', 'SPRU', 'SPWH', 'SPWRW', 'SPXX', 'SQFT', 'SQFTP', 'SQFTW', 'SQNS', 'SRAD', 'SREA', 'SRFM', 'SRG', 'SRI', 'SRJN', 'SRL', 'SRTAW', 'SRTS', 'SRV', 'SRXH', 'SRZN', 'SRZNW', 'SSACU', 'SSB', 'SSBI', 'SSD', 'SSEA', 'SSEAR', 'SSEAU', 'SSII', 'SSL', 'SSM', 'SSS', 'SSSS', 'SSSSL', 'SST', 'ST', 'STAK', 'STEW', 'STEX', 'STFS', 'STG', 'STHO', 'STI', 'STK', 'STKE', 'STKH', 'STKS', 'STM', 'STN', 'STRC', 'STRD', 'STRF', 'STRK', 'STRN', 'STRO', 'STRR', 'STRRP', 'STSS', 'STSSW', 'STTK', 'STUB', 'STVN', 'STWD', 'SUGP', 'SUIG', 'SUNC', 'SUNE', 'SUPX', 'SURG', 'SUSL', 'SUUN', 'SUZ', 'SVAC', 'SVACU', 'SVACW', 'SVAQ', 'SVAQU', 'SVAQW', 'SVCC', 'SVCCU', 'SVCCW', 'SVIVU', 'SVM', 'SVRE', 'SVREW', 'SVRN', 'SW', 'SWAG', 'SWAGW', 'SWKHL', 'SWP', 'SWVL', 'SWVLW', 'SWZ', 'SXTC', 'SXTP', 'SXTPW', 'SY', 'SYM', 'SYNX', 'SYPR', 'SZZL', 'SZZLR', 'SZZLU', 'TAC', 'TACH', 'TACHU', 'TACHW', 'TACO', 'TACOU', 'TACOW', 'TACT', 'TALKW', 'TANH', 'TAOP', 'TAOX', 'TAP.A', 'TASK', 'TATT', 'TAVI', 'TAVIR', 'TAX', 'TAXI', 'TAYD', 'TBB', 'TBBB', 'TBH', 'TBHC', 'TBLA', 'TBLAW', 'TBLD', 'TBMC', 'TBMCR', 'TBN', 'TC', 'TCBIO', 'TCBS', 'TCGL', 'TCPA', 'TCPC', 'TCRT', 'TCRX', 'TDAC', 'TDACW', 'TDAY', 'TDC', 'TDF', 'TDI', 'TDIC', 'TDOG', 'TDSC', 'TDTH', 'TDWD', 'TDWDR', 'TDWDU', 'TEI', 'TELA', 'TELO', 'TEM', 'TEN', 'TENX', 'TEO', 'TFPM', 'TFSA', 'TFSL', 'TGB', 'TGE', 'TGEN', 'TGHL', 'TGL', 'THCH', 'THG', 'THH', 'THM', 'THO', 'THQ', 'THW', 'TIER', 'TIGO', 'TIGR', 'TII', 'TIL', 'TIMB', 'TIME', 'TINY', 'TIRX', 'TISI', 'TIVC', 'TJGC', 'TKC', 'TKLF', 'TKO', 'TLF', 'TLIH', 'TLK', 'TLN', 'TLNC', 'TLNCU', 'TLNCW', 'TLPH', 'TLSA', 'TLSIW', 'TLX', 'TLYS', 'TMC', 'TMCWW', 'TMDE', 'TMH', 'TMQ', 'TMTSU', 'TMUSI', 'TMUSL', 'TMUSZ', 'TNL', 'TNMG', 'TNON', 'TNONW', 'TNYA', 'TOIIW', 'TOMZ', 'TONX', 'TOON', 'TOP', 'TOPP', 'TOPS', 'TORO', 'TOUR', 'TOVX', 'TOYO', 'TPCS', 'TPET', 'TPG', 'TPGXL', 'TPL', 'TPST', 'TPTA', 'TPVG', 'TPZ', 'TRAW', 'TRI', 'TRIB', 'TRIN', 'TRINI', 'TRINZ', 'TRMD', 'TRNR', 'TRON', 'TROO', 'TRSG', 'TRSY', 'TRT', 'TRUG', 'TRVG', 'TRX', 'TS', 'TSAT', 'TSCM', 'TSEM', 'TSI', 'TSL', 'TSLX', 'TSQ', 'TTAN', 'TTE', 'TTRX', 'TU', 'TULP', 'TUR', 'TURB', 'TVA', 'TVACU', 'TVACW', 'TVAI', 'TVAIR', 'TVAIU', 'TVC', 'TVE', 'TVGNW', 'TW', 'TWAV', 'TWFG', 'TWG', 'TWIN', 'TWLV', 'TWLVR', 'TWLVU', 'TWN', 'TWOD', 'TXMD', 'TXO', 'TY', 'TYG', 'TYGO', 'UA', 'UAE', 'UAN', 'UAVS', 'UBCP', 'UBXG', 'UCAR', 'UCL', 'UEIC', 'UFG', 'UFI', 'UG', 'UGI', 'UGRO', 'UHAL', 'UHG', 'UHGWW', 'UK', 'ULBI', 'ULS', 'ULTI', 'ULY', 'UMAC', 'UMBFO', 'UNCY', 'UNMA', 'UOKA', 'UONE', 'UONEK', 'UP', 'UPC', 'UPLD', 'UPXI', 'URG', 'USA', 'USAC', 'USAR', 'USAS', 'USBC', 'USEA', 'USEG', 'USFD', 'USG', 'USGO', 'USGOW', 'USIO', 'USMD', 'USOI', 'UTF', 'UTG', 'UTHR', 'UTSI', 'UUU', 'UWMC', 'UXIN', 'UYSC', 'UYSCR', 'UYSCU', 'UZD', 'UZE', 'UZF', 'VACH', 'VACHU', 'VACHW', 'VACI', 'VALN', 'VANI', 'VATE', 'VAVX', 'VBF', 'VBIX', 'VBNK', 'VCIC', 'VCICW', 'VCIG', 'VCV', 'VEEA', 'VEEAW', 'VEEE', 'VELO', 'VENU', 'VEON', 'VERI', 'VERU', 'VET', 'VFF', 'VFL', 'VFS', 'VFSWW', 'VG', 'VGASW', 'VGI', 'VGM', 'VGZ', 'VHC', 'VHCP', 'VHCPU', 'VHCPW', 'VHUB', 'VIA', 'VIASP', 'VIK', 'VINP', 'VIOT', 'VIRT', 'VISN', 'VIV', 'VIVS', 'VKI', 'VKQ', 'VLN', 'VLRS', 'VLT', 'VLTO', 'VLYPN', 'VLYPO', 'VLYPP', 'VMAR', 'VMO', 'VNCE', 'VNET', 'VNME', 'VNMEU', 'VNMEW', 'VNOM', 'VNRX', 'VNTG', 'VOC', 'VOD', 'VOLT', 'VOR', 'VPV', 'VRA', 'VRAX', 'VRCA', 'VRME', 'VS', 'VSA', 'VSECU', 'VSEE', 'VSEEW', 'VSME', 'VSNT', 'VSOL', 'VSTD', 'VTAK', 'VTG', 'VTGN', 'VTIX', 'VTMX', 'VTN', 'VTSI', 'VTVT', 'VTYX', 'VVOS', 'VVPR', 'VVR', 'VVV', 'VWAV', 'VWAVW', 'VYNE', 'VZLA', 'WAFDP', 'WAFU', 'WAI', 'WALDW', 'WATT', 'WAVE', 'WB', 'WBI', 'WBUY', 'WBX', 'WCC', 'WCLD', 'WCT', 'WDH', 'WDI', 'WDS', 'WEA', 'WENN', 'WENNU', 'WENNW', 'WES', 'WETH', 'WETO', 'WF', 'WFCF', 'WFF', 'WFG', 'WFRD', 'WGRX', 'WGSWW', 'WH', 'WHF', 'WHFCL', 'WHLR', 'WHLRD', 'WHLRP', 'WHR', 'WHWK', 'WIA', 'WILC', 'WIMI', 'WINN', 'WIW', 'WKEY', 'WKHS', 'WKSP', 'WLAC', 'WLACU', 'WLACW', 'WLDS', 'WLDSW', 'WLIIU', 'WLKP', 'WLTH', 'WLYB', 'WMG', 'WNW', 'WOK', 'WOOD', 'WORX', 'WPP', 'WPRT', 'WRAP', 'WRD', 'WRN', 'WSBCO', 'WSBK', 'WSHP', 'WSM', 'WSO', 'WSO.B', 'WST', 'WSTN', 'WSTNR', 'WSTNU', 'WTF', 'WTFC', 'WTFCN', 'WTG', 'WTGUR', 'WTGUU', 'WTM', 'WTO', 'WTRE', 'WU', 'WVVI', 'WVVIP', 'WW', 'WWR', 'WXM', 'WY', 'WYFI', 'WYHG', 'WYY', 'XAIR', 'XBIO', 'XBIT', 'XBP', 'XBPEW', 'XCBEU', 'XCH', 'XCUR', 'XELB', 'XELLL', 'XFLT', 'XFOR', 'XGN', 'XHG', 'XHLD', 'XIDE', 'XIFR', 'XLO', 'XNET', 'XOMAO', 'XOMAP', 'XONE', 'XOS', 'XOSWW', 'XP', 'XPL', 'XPON', 'XRPC', 'XRPN', 'XRPNU', 'XRPNW', 'XRTX', 'XRXDW', 'XSLLU', 'XT', 'XTIA', 'XTKG', 'XTLB', 'XTNT', 'XWEL', 'XWIN', 'XXI', 'XYF', 'XYZ', 'XZO', 'YAAS', 'YALA', 'YB', 'YCBD', 'YCY', 'YDDL', 'YDES', 'YDESW', 'YDKG', 'YHC', 'YHGJ', 'YHNA', 'YHNAR', 'YHNAU', 'YI', 'YIBO', 'YMAT', 'YMM', 'YMT', 'YOUL', 'YQ', 'YRD', 'YSG', 'YSS', 'YSXT', 'YTRA', 'YXT', 'YYAI', 'YYGH', 'ZAP', 'ZBAI', 'ZBAO', 'ZCMD', 'ZDAI', 'ZDGE', 'ZENA', 'ZENV', 'ZEO', 'ZEOWW', 'ZEPP', 'ZG', 'ZGM', 'ZH', 'ZIONP', 'ZJK', 'ZJYL', 'ZKH', 'ZKIN', 'ZKP', 'ZKPU', 'ZKPW', 'ZNB', 'ZONE', 'ZOOZ', 'ZOOZW', 'ZSTK', 'ZTEK', 'ZTR', 'ZURA', 'ZYBT', 'ALCYU', 'BAYAU', 'CAPNU', 'CHARU', 'CLCO', 'DRDBU', 'DTSQU', 'EFTY', 'EGHAU', 'EMPG', 'EURKU', 'GPATU', 'GSHRU', 'JMG', 'KVACU', 'LAWR', 'MAGH', 'MAMK', 'MCTA', 'NOEMU', 'NUTR', 'NXC', 'NXN', 'OAKUU', 'OST', 'PC', 'PCL', 'PCTTU', 'PLTS', 'POLEU', 'PTNM', 'QETAU', 'QMMM', 'RFAIU', 'RIBBU', 'ROC', 'SDM', 'SPKLU', 'SVA', 'TAVIU', 'TDACU', 'VCICU', 'WHLRL', 
  
   

]

PERIODO_ANOS_ENTRENAMIENTO = 3          # años de historia para cada loop
FWD_DAYS = 20                           # días hacia adelante p/target de entrenamiento (clasificador)
FWD_DAYS_TARGET = 7                     # días hacia adelante p/retorno “una semana”
N_SEMANAS_BACKTEST = 1                 # ← por ahora 5; si exporta ok, subimos a 100
OUTPUT_XLSX = Path("Consolidado_1_semanas_todos los tickers_3.xlsx")
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
