# system
import os,sys
from datetime import datetime, date, timedelta
import time
import multiprocessing
import traceback
import pathlib
import zipfile
from contextlib import contextmanager

# spatial
from rasterio.session import AWSSession

import rasterio
from rasterio.io import MemoryFile, ZipMemoryFile
import rasterio.mask
from rasterio.windows import Window
import pyproj, shapely
import sentinelsat
import pandas as pd
import geopandas as gpd
from eolearn.core import FeatureType, EOPatch, LoadTask, OverwritePermission
from eolearn.io import ExportToTiff
from sentinelhub import BBoxSplitter, CRS, BBox

# Data structures
import pandas as pd
import numpy as np

from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *

# AWS access
import boto3

class s2Dwnld():
    def __init__(self, username, pswrd, gridList, timeQuery, maxcc):
        '''
        Add variable descriptions you lazy bastard

        '''
        self.username = username
        self.pswrd = pswrd
        self.gridList = gridList
        self.timeQuery = timeQuery
        self.api = sentinelsat.SentinelAPI(self.username, self.pswrd, 
                                'https://scihub.copernicus.eu/dhus', 
                                show_progressbars=False)
        #self.api = sentinelsat.SentinelAPI(self.username, self.pswrd, 
        #                        'https://scihub.copernicus.eu/apihub/', 
        #                        show_progressbars=False)
        # validating sentinel hub account
        #--------------------------------
        try:
            loginTest = self.api.query(tileid = '34HCH', date = ('20190101', date(2019,1, 20)),
                                        platformname='Sentinel-2',
                                        cloudcoverpercentage=(0, 80),
                                        producttype = 'S2MSI1C')
        except sentinelsat.sentinel.SentinelAPIError:
            raise Exception('::S2DWNLD-ERROR:: User name or password not correct,\n \
                            create a valid account from https://scihub.copernicus.eu/dhus')
        
        # validating s2 grid blocks
        #--------------------------
        try:
            girdTest = self.gridList[0]
        except Exception:
            raise Exception('::S2DWNLD-ERROR:: S2 grid names are not provided in list format')
        s2grid = loadGrid()
        gridNameList = s2grid['name'].to_numpy()
        for passedGrid in self.gridList:
            if passedGrid not in gridNameList:
                raise Exception('::S2DWNLD-ERROR:: Provided S2-gird id is not valid')   
        s2grid = s2grid[s2grid['name'].isin(self.gridList)]
        self.s2grid = s2grid

        # validating start and end date
        #------------------------------
        try:
            sd_list = timeQuery[0].split('-')
            startDate = date(int(sd_list[0]), 
                             int(sd_list[1]), 
                             int(sd_list[2]))
        
            ed_list = timeQuery[1].split('-')
            endDate = date(int(ed_list[0]), 
                           int(ed_list[1]), 
                           int(ed_list[2]))
        except Exception:
            tr = traceback.format_exc()
            raise Exception('::S2DWNLD-ERROR:: Date format is wrong please provide the\n \
                             start and end date in the following format:\n \
                             ["yyyy-mm-dd", "yyyy-mm-dd"]')
        self.startDate = startDate
        self.endDate = endDate

        # validating cloudcover
        #----------------------
        try:
            self.maxcc = int(maxcc)
            if self.maxcc > 100 or self.maxcc < 0:
                raise Exception('::S2DWNLD-ERROR:: maxcc value is not valid [<100 and > 0]')
        except Exception:
            raise Exception('::S2DWNLD-ERROR:: maxcc value is not valid (not a number)')

def loadGrid():
    __location__ = os.path.realpath(os.path.join(os.getcwd(), 
                                    os.path.dirname(__file__)))
    gridpath = os.path.join(__location__, 'resources\S2_mgrs_world.shp')
    s2Grid = gpd.read_file(gridpath)
    #s2Grid = pd.read_pickle(gridpath)
    #s2Grid = gpd.GeoDataFrame(s2Grid)
    #s2Grid.crs = {'init': 'epsg:4326'}
    return s2Grid


def initiateLog(logDir, s2obj):
    # Intiating log file header
    timestamp = datetime.now().strftime("%H%M%S_%Y%m%d")
    logName = 'S2dwnld_'+ timestamp+\
              '_logfile.txt'
    file = open(logDir/logName, "w+") 
    file.write("Log file for S2 downloads, time: " +timestamp+'\n')
    file.write('------------------------------------------------\n')
    file.write('sci-hub username: '+s2obj.username+'\n')
    file.write('sci-hub password: '+s2obj.pswrd+'\n')
    file.write('For time range: '+s2obj.timeQuery[0]+' to '+
                s2obj.timeQuery[1]+'\n')
    file.write('For S2 Grid blocks: ')
    for gridid in s2obj.gridList:
        file.write(' '+ gridid + ' ')
    file.write('\n')
    file.write('Cloud cover percentage: '+ str(s2obj.maxcc)+'\n')
    file.write('------------------------------------------------\n')
    file.close()
    return logDir/logName

def writeLog(logfile,message):
    #For if the file is locked
    retry_idx = 0
    retry = True 
    while retry == True and retry_idx < 3:
        try:
            with open(logfile, "a") as f:
                f.write(message+"\n")
                f.close()
            retry=False
        except Exception:
            time.sleep(2)
            retry_idx+=1

def dowloadToPath(s2obj, dfRow, outpath, logfile, return_dict):   
    '''
    query   ::  sentinelsat object, result of the api query

    outpath ::  string with output path
                e.g 'C:/S2dwnld/'
    '''
    try:
        t0 = time.time()
        #s2obj.api.download_all(query.index, directory_path= outpath)

        downloadID = dfRow.name
        s2obj.api.download_all([downloadID], directory_path= outpath)
        return_dict['status'] = True
        logmsg = dfRow['title'] +\
                 ' download done: '+str(round((time.time()-t0)/60,2))+'\n'
        writeLog(logfile,logmsg)
    #Still need to refine error    
    except Exception as err:
        tr = traceback.format_exc()
        return_dict['status'] = False
        return_dict['error'] = tr
        sys.exit()

def daterange(start_date, end_date):
        '''
        start_date  ::  date object of start date
                        e.g. date(2019,1,1)
        end_date    ::  date object of end data
                        e.g. date(2019,2,25)
        '''
        for n in range(int ((end_date - start_date).days)):
            yield start_date + timedelta(n)

@contextmanager
def mem_raster(data, **profile):
    with MemoryFile() as memfile:
        with memfile.open(**profile) as dataset_writer:
            dataset_writer.write(data,1)
 
        with memfile.open() as dataset_reader:
            yield dataset_reader
 
def savePatch(eoPatch, foldername):

    eoPatch.save(foldername, 
                 overwrite_permission=OverwritePermission.OVERWRITE_FEATURES, 
                 compress_level=7)


def translateToPatches(rasMemList, eoDir, gridID, dims, S2Name):
    '''
    Funtion that loops through the blocks and creates eo pathces
    '''
    mp = multiprocessing.Pool(processes = 5)
    
    heightOff10=0
    widthOff10=0
    heightOff20=0
    widthOff20=0

    hBS10=dims[0]/10
    wBS10=dims[1]/10
    hBS20=dims[2]/10
    wBS20=dims[3]/10
    
    eop_number=1
    for i in range(10):
        widthOff10 = 0
        widthOff20 = 0
        for j in range(10):
            # Convert bands for block into eopatch format
            initiate10 = False
            initiate20 = False
            for raster in rasMemList:
                if int(raster[2]) == 10:
                    nowWindow = Window(int(widthOff10), int(heightOff10), int(wBS10), int(hBS10))
                    
                    profile ={'driver':'GTiff',
                                'height':raster[0].shape[0], 
                                'width':raster[0].shape[1], 
                                'count':1, 
                                'dtype':str(raster[0].dtype),
                                'proj':raster[1]['crs'], 
                                'transform':raster[3]
                              }
 
                    with mem_raster(raster[0], **profile) as ds:
                        nowBlock = ds.read(1, window=nowWindow)
                        win_transform = list(ds.window_transform(nowWindow))
 
                    
                    if initiate10 == False:
                        bands10 = nowBlock[np.newaxis, :,:,np.newaxis]
                        bbox10 = BBox(bbox=[win_transform[2], 
                                            win_transform[5], 
                                            win_transform[2]+(win_transform[0]*nowBlock.shape[0]),
                                            win_transform[5]-(win_transform[0]*nowBlock.shape[1])],
                                            crs = str(raster[1]['crs']))
                        metaBandNames10 = [raster[4]]
                        initiate10 = True

                    else:
                        bands10 = np.append(bands10, nowBlock[np.newaxis, :,:, np.newaxis], axis=3)
                        metaBandNames10.append(raster[4])              
                    
                elif int(raster[2]) == 20:
                    nowWindow = Window(int(widthOff20), int(heightOff20), int(wBS20), int(hBS20))

                    profile ={'driver':'GTiff',
                                'height':raster[0].shape[0], 
                                'width':raster[0].shape[1], 
                                'count':1, 
                                'dtype':str(raster[0].dtype),
                                'proj':raster[1]['crs'], 
                                'transform':raster[3]
                              }
 
                    with mem_raster(raster[0], **profile) as ds:
                        nowBlock = ds.read(1, window=nowWindow)
                        win_transform = list(ds.window_transform(nowWindow))
                    
                    if initiate20 == False:
                        bands20 = nowBlock[np.newaxis, :,:, np.newaxis]
                        metaBandNames20 = [raster[4]]
                        initiate20 = True
                        if raster[4] == 'SCL':
                            '''
                            SCL layer
                            1	Saturated or defective
                            2	Dark Area Pixels
                            3	Cloud Shadows
                            4	Vegetation
                            5	Bare Soils
                            6	Water
                            7	Clouds Low Probability / Unclassified
                            8	Clouds Medium Probability
                            9	Clouds High Probability
                            10	Cirrus
                            11	Snow / Ice
                            '''
                            mask20 = np.where(((nowBlock == 1) | 
                                              (nowBlock == 2) |
                                              (nowBlock == 3) |
                                              (nowBlock == 8) |
                                              (nowBlock == 9) |
                                              (nowBlock == 10) |
                                              (nowBlock == 11)), 1,0)

                    else:
                        bands20 = np.append(bands20, nowBlock[np.newaxis, :,:, np.newaxis], axis=3)
                        metaBandNames20.append(raster[4])
                        if raster[4] == 'SCL':
                            mask20 = np.where(((nowBlock == 1) | 
                                              (nowBlock == 2) |
                                              (nowBlock == 3) |
                                              (nowBlock == 8) |
                                              (nowBlock == 9) |
                                              (nowBlock == 10) |
                                              (nowBlock == 11)), 1,0)

            str_eopnr = str(eop_number)
            for i in range(3-len(str(eop_number))):
                str_eopnr = '0'+str_eopnr

            eopName = gridID+'_'+str_eopnr+"_eopatch_"+\
                                 str(bbox10).replace(',','_')+\
                                 '_'+str(raster[1]['crs']).replace(':', '_')
            
            if not (eoDir/eopName).exists():    # if patch does not exist yet - Create new eopatch
                nowPatch = EOPatch()
                nowPatch[FeatureType.DATA]['BANDS10'] = bands10
                nowPatch[FeatureType.DATA]['BANDS20'] = bands20
                nowPatch[FeatureType.MASK]['CLM']= mask20[np.newaxis,:,:, np.newaxis]
                nowPatch[FeatureType.META_INFO]['BAND_NAMES10'] = metaBandNames10
                nowPatch[FeatureType.META_INFO]['BAND_NAMES20'] = metaBandNames20
                nowPatch[FeatureType.META_INFO]['s2_scene_ID'] = [S2Name]
                nowPatch[FeatureType.BBOX] = bbox10
                nowPatch[FeatureType.TIMESTAMP] = [datetime.strptime(S2Name[11:19], "%Y%m%d").date()]
    
                #nowPatch.save(str(eoDir/eopName), 
                #              overwrite_permission=OverwritePermission.OVERWRITE_FEATURES, 
                #              compress_level=5)
                mp.apply_async(savePatch, args=(nowPatch, str(eoDir/eopName)))
                #nowPatch = bands10 = bands20 = mask20 = None # Clear memory
            else:   # it exists, so only have to load (#### TO DO ### see if it necissary to load whole patch to add to it)
                #loadNow = LoadTask(eoDir)
                #nowPatch = loadNow.execute(eopName)
                nowPatch = EOPatch.load(str(eoDir/eopName))

                nowPatch[FeatureType.DATA]['BANDS10'] = np.append(nowPatch[FeatureType.DATA]['BANDS10'],
                                                                  bands10, axis=0)
                nowPatch[FeatureType.DATA]['BANDS20'] = np.append(nowPatch[FeatureType.DATA]['BANDS20'],
                                                                  bands20, axis=0)
                nowPatch[FeatureType.MASK]['CLM']= np.append(nowPatch[FeatureType.MASK]['CLM'],
                                                             mask20[np.newaxis,:,:, np.newaxis], axis=0) 
                nowPatch[FeatureType.META_INFO]['s2_scene_ID'] = nowPatch[FeatureType.META_INFO]['s2_scene_ID']+[S2Name]
                nowPatch[FeatureType.TIMESTAMP] = nowPatch[FeatureType.TIMESTAMP]+\
                                                  [datetime.strptime(S2Name[11:19], "%Y%m%d").date()]
                #nowPatch.save(str(eoDir/eopName), 
                #              overwrite_permission=OverwritePermission.OVERWRITE_FEATURES, 
                #              compress_level=5)
                mp.apply_async(savePatch, args=(nowPatch, str(eoDir/eopName)))
                #nowPatch = bands10 = bands20 = mask20 = None # Clear memory

            eop_number+=1
            widthOff10+=wBS10
            widthOff20+=wBS20
        heightOff10+=hBS10
        heightOff20+=hBS20
    
    mp.close()
    mp.join()
    


def createPatches(s2obj, patchSavePath, tempDir=None, _callback=None):
    '''
    s2obj           ::  Object of class s2Dwnld

    patchSavePath   ::  string of directory, where the downloaded eopatches will be saved
                        e.g. 'C:/S2dwnld/'
    tempDir         ::  (optional) if the downloaded file sould be saved in a temp dir (e.g. SD drive)
                        activating this variable will temporarily process the download in this folder
                        if None everything is processed in the patchSavePath
    '''
    if tempDir is not None:
        workspace = tempDir
    else:
        workspace = patchSavePath
    workspace = pathlib.Path(workspace)
    if not workspace.exists():
        raise Exception('::S2DWNLD-ERROR:: Specified folder does not exist')
    eopDir = pathlib.Path(patchSavePath)
    if not eopDir.exists():
        raise Exception('::S2DWNLD-ERROR:: Specified folder does not exist')

    startDate = s2obj.startDate
    endDate = s2obj.endDate
    gridList = s2obj.gridList
    
    #Initial logfile
    logDir = eopDir
    logfile = initiateLog(logDir, s2obj)
    
    # Set workflow manager dictionaty
    workflow_status=dict()
    workflow_status['toProcess'] = False
    for gridId in gridList:
        gridT0 = time.time()
        print('For: ', gridId)
        # if directory for this grid does not exist yet
        if not (eopDir/gridId).exists():
            (eopDir/gridId).mkdir()
            eopDir = eopDir/gridId
        else:
            eopDir = eopDir/gridId

        # geometry of current grid
        AOI = s2obj.s2grid.loc[s2obj.s2grid['name'] == gridId].iloc[0]['geometry']
        totalDays = sum(1 for _ in daterange(startDate, endDate))

        # Getting total amount of S-2 images here for callback function
        totalDateQuery = (startDate.strftime("%Y%m%d"), endDate)
        totalS2 = s2obj.api.query(AOI.wkt,
                                  date = totalDateQuery,
                                  platformname='Sentinel-2',
                                  cloudcoverpercentage=(0, s2obj.maxcc),
                                  producttype = 'S2MSI2A')
        totalS2DF =  s2obj.api.to_dataframe(totalS2)
        if totalS2DF.empty:
            break 
        totalS2DF['GridName'] = totalS2DF['title'].map(lambda x: x.split("_")[5][1:])
        totalS2oneGridDF = totalS2DF.loc[totalS2DF['GridName'] == gridId]
        if totalS2oneGridDF.empty:
            break
        totalDwnld = totalS2oneGridDF.shape[0]

        #for single_date in daterange(startDate, endDate):
        callback_cntr=1
        for idx, row in totalS2oneGridDF.iterrows():
        
            retry_idx = 0
            retry = True
            while retry == True and retry_idx < 3:
                # Setup and perform query
                '''
                nowDateQuery = (single_date.strftime("%Y%m%d"), single_date + timedelta(days=1))
                nowImageQuery = s2obj.api.query(AOI.wkt,
                                                date = nowDateQuery,
                                                platformname='Sentinel-2',
                                                cloudcoverpercentage=(0, s2obj.maxcc),
                                                producttype = 'S2MSI2A')
                                                #'S2MSI1C', 'S2MSI2A'
                nowImageQueryDF =  s2obj.api.to_dataframe(nowImageQuery)
                '''
                # If no imagery is available
                '''
                if nowImageQueryDF.empty:
                    retry = False
                    break
                nowImageQueryDF['GridName'] = nowImageQueryDF['title']\
                                                .map(lambda x: x.split("_")[5][1:])
                oneGrid = nowImageQueryDF.loc[nowImageQueryDF['GridName'] == gridId]
                '''
                # If specific block is not avaialbe
                '''
                if oneGrid.empty:
                    retry = False
                    break
                else:
                '''
                #S2Name = oneGrid['title'].iloc[0]        
                S2Name = row['title']      
                # Check if the current image has not already been downloaded
                existingPathces = os.listdir(eopDir)
                if len(existingPathces)!= 0:
                    for tp in existingPathces:
                        if tp.split('_')[2]=='eopatch':
                            testpatch = tp
                            break
                    tPatch = EOPatch.load(str(eopDir/testpatch), features=FeatureType.META_INFO)
                    doneScenes= tPatch[FeatureType.META_INFO]['s2_scene_ID']
                    if S2Name in doneScenes:
                        workflow_status['s2scene'] = 'done'
                        logmsg = S2Name+ ' already processed\n'
                        writeLog(logfile,logmsg)
                        retry=False
                    else:
                        workflow_status['s2scene'] = 'notDone'
                else:
                    workflow_status['s2scene'] = 'notDone'

                if workflow_status['s2scene'] == 'notDone':   # If the current s2 scene has not been downloaded          
                    
                    # Start mutiprocessing to download
                    manager = multiprocessing.Manager()
                    return_dict = manager.dict()

                    p = multiprocessing.Process(target=dowloadToPath, 
                                                args=(s2obj, 
                                                    row,
                                                    workspace,
                                                    logfile,
                                                    return_dict))
                    p.start()

                    # While waiting for another download, start to process the already downloaded file
                    if workflow_status['toProcess'] == True: ### HERE

                        procT0= time.time()
                        S2proc =  workflow_status['toProcessS2ID']
                        completeFile = workspace / (S2proc+".zip")
                        if completeFile.is_file() == False:
                            raise Exception('::S2DWNLD-ERROR:: Somethings wrong, could not find dowloaded file')
                        m10bands = ['_B02_10m.jp2', '_B03_10m.jp2', '_B04_10m.jp2', '_B08_10m.jp2']
                        m20bands = ['_B05_20m.jp2', '_B06_20m.jp2', '_B07_20m.jp2', '_B8A_20m.jp2', 
                                    '_B11_20m.jp2', '_B12_20m.jp2', '_SCL_20m.jp2']
                        
                        with zipfile.ZipFile(completeFile) as zip:
                            allfiles = zip.namelist()
                        nameListdf = pd.DataFrame(allfiles, columns=['zipPathNames'])
                        nameListdf['suffix'] = nameListdf['zipPathNames'].map(lambda x: str(x)[-12:])
                        nameListdf = nameListdf.sort_values('zipPathNames')
                        zipPathNames = nameListdf.loc[(nameListdf['suffix'].isin(m10bands))\
                                                    | (nameListdf['suffix'].isin(m20bands))]\
                                                    ['zipPathNames'].to_list()
                        
                        # load bands into memory
                        #----------------------
                        with open(completeFile, 'rb') as f, ZipMemoryFile(f) as memzipfile:
                            bands=[]
                            dim10 = False
                            dim20=False
                            for loadraster in zipPathNames:
                                with memzipfile.open(loadraster, driver = 'JP2OpenJPEG') as src:
                                    bands.append([src.read(1), src.profile, 
                                                list(src.profile['transform'])[0], src.transform,
                                                loadraster[-11:-8]])
                                    if int(list(src.profile['transform'])[0]) == 10 and dim10==False:
                                        h10 = src.profile['height']
                                        w10 = src.profile['width']
                                        dim10=True
                                    if int(list(src.profile['transform'])[0]) == 20 and dim20==False:
                                        h20 = src.profile['height']
                                        w20 = src.profile['width']
                                        dim20=True 
                        translateToPatches(bands, eopDir, gridId, [h10,w10,h20,w20], S2proc)
                        workflow_status['toProcess'] = False

                        logmsg = S2proc+ ' processing done: '+\
                                        str(round((time.time()-procT0)/60,2))+'\n'
                        writeLog(logfile,logmsg)

                        # CALLBACK HERE
                        _callback.emit((callback_cntr/totalDwnld)*100)
                        callback_cntr+=1

                        # Delete downloaded zip file
                        os.remove(completeFile)

                    p.join(900)     # Wait atleast 900 seconds for download, otherwise ussume it froze
                    if p.is_alive():
                        p.terminate()
                        p.join()
                        incompleteFile = workspace / str(S2Name+".zip.incomplete")
                        if incompleteFile.is_file():
                            os.remove(incompleteFile)
                        logmsg = S2Name+ ' download froze, rerty['+str(retry_idx)+']\n'
                        writeLog(logfile,logmsg)
                        retry_idx+=1

                    elif return_dict['status'] == False:
                        incompleteFile = workspace / str(S2Name+".zip.incomplete")
                        if incompleteFile.is_file():
                            os.remove(incompleteFile) 
                        logmsg = S2Name+ ' download failed, rerty['+str(retry_idx)+']'+\
                                        return_dict['error']+'\n'
                        writeLog(logfile,logmsg)
                        retry_idx+=1
                        
                    else:
                        retry=False
                        workflow_status['toProcess'] = True
                        workflow_status['toProcessS2ID'] = S2Name
                         
            if retry == True:
                logmsg = S2Name+ ' **ALL RETRIES FAILED**\n'
                writeLog(logfile,logmsg)

        # Do last processing here
        if workflow_status['toProcess'] == True: ### HERE

            procT0= time.time()
            S2proc =  workflow_status['toProcessS2ID']
            completeFile = workspace / (S2proc+".zip")
            if completeFile.is_file() == False:
                raise Exception('::S2DWNLD-ERROR:: Somethings wrong, could not find dowloaded file')
            m10bands = ['_B02_10m.jp2', '_B03_10m.jp2', '_B04_10m.jp2', '_B08_10m.jp2']
            m20bands = ['_B05_20m.jp2', '_B06_20m.jp2', '_B07_20m.jp2', '_B8A_20m.jp2', 
                        '_B11_20m.jp2', '_B12_20m.jp2', '_SCL_20m.jp2']
            
            with zipfile.ZipFile(completeFile) as zip:
                allfiles = zip.namelist()
            nameListdf = pd.DataFrame(allfiles, columns=['zipPathNames'])
            nameListdf['suffix'] = nameListdf['zipPathNames'].map(lambda x: str(x)[-12:])
            nameListdf = nameListdf.sort_values('zipPathNames')
            zipPathNames = nameListdf.loc[(nameListdf['suffix'].isin(m10bands))\
                                        | (nameListdf['suffix'].isin(m20bands))]\
                                        ['zipPathNames'].to_list()
            
            # load bands into memory
            #----------------------
            with open(completeFile, 'rb') as f, ZipMemoryFile(f) as memzipfile:
                bands=[]
                dim10 = False
                dim20=False
                for loadraster in zipPathNames:
                    with memzipfile.open(loadraster, driver = 'JP2OpenJPEG') as src:
                        bands.append([src.read(1), src.profile, 
                                    list(src.profile['transform'])[0], src.transform,
                                    loadraster[-11:-8]])
                        if int(list(src.profile['transform'])[0]) == 10 and dim10==False:
                            h10 = src.profile['height']
                            w10 = src.profile['width']
                            dim10=True
                        if int(list(src.profile['transform'])[0]) == 20 and dim20==False:
                            h20 = src.profile['height']
                            w20 = src.profile['width']
                            dim20=True 
            translateToPatches(bands, eopDir, gridId, [h10,w10,h20,w20], S2proc)
            workflow_status['toProcess'] = False

            logmsg = S2proc+ ' processing done: '+\
                            str(round((time.time()-procT0)/60,2))+'\n'
            writeLog(logfile,logmsg)

            #CALLBACK HERE
            _callback.emit((callback_cntr/totalDwnld)*100)
            callback_cntr+=1

            # Delete downloaded zip file
            os.remove(completeFile)

        logmsg = gridId+ ' Done All: '+\
                 str(round((time.time()-gridT0)/60,2))+'\n'+\
                '------------------------------------------------\n'
        writeLog(logfile,logmsg)


def mock_function(total, _callback=None):

    print('In here...')
    for i in range(1,total+1):
        print('sleeping...')
        time.sleep(3)
        if _callback:
            #dlg.updateProgress((i/total)*100)
            _callback.emit((i/total)*100)
    
# QT multithreading worker functions
# ----------------------------------
            
class WorkerSignals(QObject):

    finished = pyqtSignal()
    error = pyqtSignal(tuple)
    result = pyqtSignal(object)
    progress = pyqtSignal(int)

class Worker(QRunnable):
    
    def __init__(self, fn, *args, **kwargs):
            super(Worker, self).__init__()

            # Store constructor arguments (re-used for processing)
            self.fn = fn
            self.args = args
            self.kwargs = kwargs
            self.signals = WorkerSignals()    

            # Add the callback to our kwargs
            self.kwargs['progress_callback'] = self.signals.progress        

    @pyqtSlot()
    def run(self):
        '''
        Initialise the runner function with passed args, kwargs.
        '''
        
        # Retrieve args/kwargs here; and fire processing using them
        try:
            result = self.fn(*self.args, **self.kwargs)
        except:
            traceback.print_exc()
            exctype, value = sys.exc_info()[:2]
            self.signals.error.emit((exctype, value, traceback.format_exc()))
        else:
            self.signals.result.emit(result)  # Return the result of the processing
        finally:
            self.signals.finished.emit()  # Done


# QT ui
# ----------------------------------

class CustomDialog(QDialog):

    def __init__(self, userName, pswrd, s2Grid, startDate, endDate, maxcloud):
        #super(CustomDialog, self).__init__(*args, **kwargs)
        super(CustomDialog, self).__init__()

        self.setWindowTitle('Downloading S2 ...')

        self.userName = userName
        self.pswrd = pswrd
        self.s2Grid = s2Grid
        self.startDate = startDate
        self.endDate = endDate
        self.maxcloud = maxcloud
        
        self.s2dnldInfo = QLabel('Downloading for grid tile: '+self.s2Grid+'\n'\
                                  'From: ['+self.startDate + '] to ['+\
                                  self.endDate+'] with maxcc: '+str(self.maxcloud))

        self.S2progress = QProgressBar()
        self.S2progress.setVisible(True)
        self.S2progress.setValue(0)

        self.StartDownload = QPushButton("Start S2-Download")
        
        self.okbutton = QPushButton("OK")
        self.cancelButton =QPushButton("Exit")
        
        self.hboxlayout_final = QHBoxLayout()
        #self.hboxlayout_final.addWidget(self.okbutton)
        self.hboxlayout_final.addWidget(self.cancelButton)
        
        self.layout = QVBoxLayout()
        self.layout.addWidget(self.s2dnldInfo)
        self.layout.addWidget(self.StartDownload)
        self.layout.addWidget(self.S2progress)
        self.layout.addLayout(self.hboxlayout_final)
        self.setLayout(self.layout)

        self.okbutton.clicked.connect(self.on_accept)
        self.cancelButton.clicked.connect(self.on_reject)
        self.StartDownload.clicked.connect(self.spinThread)

        self.threadpool = QThreadPool()

    def runS2dwnld(self):
        mock_function(4)

    def progress_fn(self, amount):
        #update progressbar here
        self.S2progress.setValue(amount)

    def thread_complete(self):
        text = self.s2dnldInfo.text()
        self.s2dnldInfo.setText(text + '\n ##COMPLETED##')

    
    def spinThread(self):
        self.StartDownload.setEnabled(False)

        worker = Worker(self.execute_this_fn)
        worker.signals.result.connect(self.print_output)
        worker.signals.finished.connect(self.thread_complete)
        worker.signals.progress.connect(self.progress_fn)
        # Execute
        self.threadpool.start(worker)

    def execute_this_fn(self, progress_callback):
        # Download function here

        #progress_callback.emit(s2api.createPatches(dwnld, 'E:/1_Geography_Systems/eopatches',tempDir= 'C:/S2dwnld/',_callback=progress_callback))
        #progress_callback.emit(mock_function(4, _callback=progress_callback))
        dwnldObj = s2Dwnld(self.userName, self.pswrd, [self.s2Grid], [self.startDate, self.endDate], self.maxcloud)
        progress_callback.emit(createPatches(dwnldObj, 'E:/1_Geography_Systems/eopatches','C:/S2dwnld/', _callback=progress_callback))

        return "Done."
    
    def print_output(self, s):
        print(s)

    def on_accept(self):
        self.accept()

    def on_reject(self):
        self.reject()


def list_objects(client, bucket, prefix=None):
    """AWS s3 list objects."""

    paginator = client.get_paginator('list_objects_v2')

    files = []
    if prefix:
        for subset in paginator.paginate(Bucket=bucket, Prefix=prefix):
            files.extend(subset.get("Contents", []))
    else:
        for subset in paginator.paginate(Bucket=bucket): 
            files.extend(subset.get("Contents", []))   
            

    return [r["Key"] for r in files]


if __name__ == '__main__':

    
    s3_client = boto3.Session().client('s3',
                                       region_name='us-west-2')

    bucket_names={'s2_meta':'sentinel-cogs-inventory',
                  's2_data':'sentinel-cogs',
                  'no2':'omi-no2-nasa'}

    #files = list_objects(s3_client, bucket_names['no2'], prefix= 'OMI-Aura_L3')
    prefix = 'sentinel-s2-l2a-cogs/2020/S2A_34HDJ'
    files = list_objects(s3_client, bucket_names['s2_data'], prefix=prefix)
    #files_2019 = list(filter(lambda x: x.split("_")[2][0:4] == "2019", files))
    print('Nice')
    print(len(files))
    for file in files[0:10]:
        print(file)

    print('Attempting download')
    # create AWS session object
    aws_session = AWSSession(boto3.Session())
    with rasterio.Env(aws_session):
        with rasterio.open('s3://'+bucket_names['s2_data']+'/sentinel-s2-l2a-cogs/2020/S2A_34HDJ_20200103_0_L2A/B02.tif') as src:
            profile = src.profile
            arr = src.read(1)

    with rasterio.open(r'E:\1_Geography_Systems\eopatches\S2A_34HDJ_20200103_0_L2A_B02.tif', 'w', **profile) as dst:
            dst.write(arr, 1)

    print('Done check')

    
    
    '''
    response = s3_client.get_object(Bucket='sentinel-cogs-inventory', 
                                    Key='tiles/7/W/FR/2018/3/31/0/B01.jp2'
                                    RequestPayer='requester')
    response_content = response['Body'].read()
    print(response_content)
    '''
    sys.exit()

    # with open('./B01.jp2', 'wb') as file:
        # file.write(response_content)
    
    
    '''
    userName = sys.argv[1]
    pswrd = sys.argv[2]
    s2Grid = sys.argv[3]
    startDate = sys.argv[4]
    endDate = sys.argv[5]
    maxcloud = int(sys.argv[6])
    '''
    userName = 'jaschamuller'
    pswrd = 'ESAOPENCLOUD'
    s2Grid = '34HBH'
    startDate = '2020-01-01'
    endDate = '2020-01-31'
    maxcloud = 80

    app = QApplication([])
    dlg = CustomDialog(userName = userName, 
                        pswrd= pswrd,
                        s2Grid=s2Grid,
                        startDate =startDate,
                        endDate =endDate,
                        maxcloud=maxcloud)
    dlg.show()

    if dlg.exec_():
        print("Success!")
    else:
        print("Cancel!")
    
    '''    
    # Create download object
    userName = 'jaschamuller'
    pswrd = 'ESAOPENCLOUD'
    dwnld = s2Dwnld(userName, pswrd, ['34HCH'], ['2019-01-01', '2019-03-01'], 80)

    #createPatches(s2obj, patchSavePath, tempDir=None)
    createPatches(dwnld, 'E:/1_Geography_Systems/eopatches','C:/S2dwnld/')

    print('DONE ALL')
    '''
