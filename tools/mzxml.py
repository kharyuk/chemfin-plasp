# ##############################################################################
# Routines for compressing and decompressing mzXML files.
# We propose the following way of compressing:
# mzXML -> python dictionary -> binary bytes (pickle) -> lzma archive
# 
# Required packages:
# * numpy ( https://www.numpy.org/ )
# * pylzma ( https://github.com/fancycode/pylzma )
#
# ##############################################################################

try:
    import xml.etree.cElementTree as ET
except:
    import xml.etree.ElementTree as ET

try:
    import cPickle
except:
    import Pickle as cPickle

import numpy as np
import xml.dom.minidom as xmlmd
import base64
import struct
import pylzma
import os
import itertools

# global variables (xml correction)
_HEAD_INCORRECT = '<?xml version="1.0" ?>'
_HEAD_CORRECT = '<?xml version="1.0" encoding="UTF-8" standalone="no"?>'


def mzXML2dict(fname):
    '''
    Convert mzXML file (fname) into python dictionary.
    Returns a <<tree-like>> dictionary.
    '''
    tree = ET.ElementTree(file=fname)
    root = tree.getroot()
    nmsp = root.tag.replace('{', '')
    nmsp = nmsp.split('}')
    namespace = {nmsp[1]: nmsp[0]}
    rootNode = {}
    rootNode['namespace'] = namespace
    rootNode['attrib'] = root.attrib
    children = root.getchildren()
    if len(children) > 0:
        rootNode['children'] = parseTree(children, namespace)
    else:
        rootNode['children'] = {}
    return rootNode

# function getMZIpair() : thanks to Taejoon Kwoon
# https://groups.google.com/forum/#!topic/spctools-discuss/qK_QThoEzeQ
def getMZIpair(string):
    '''
    Extract M/Z and intensity arrays from bytearray (string) in mzXML file.
    Returns two numpy arrays (mz, intensity)
    '''
    procString = base64.decodestring(string)
    size = len(procString)/4
    unFormat = '>%dL' % size
    ind = 0
    intensity = []
    mz = []
    for element in struct.unpack(unFormat, procString):
        iElement = struct.pack('I', element)
        fElement = struct.unpack('f', iElement)[0]
        # fElement = float(fElement)
        if ((ind % 2) == 0):
            mz.append(fElement)
        else:
            intensity.append(fElement)
        ind += 1
    mz = np.array(mz)
    intensity = np.array(intensity)
    return mz, intensity

def numerize(x):
    '''
    Tries to convert x into integer, then into float number. Preserves original
    input if there is no success. 
    '''
    try:
        y = int(x)
    except:
        try:
            y = float(x)
        except:
            y = x
    return y

def parseTree(listXML, namespace, use_numerize=False):
    '''
    XML tree parser. Routine used for conversion from mzXML to dictionary.
    
    Input parameters:
        listXML - list with xml subtrees.
        namespace - dictionary containing specific namesace of file
        use_numerize (=False) - do int/float conversion where it is possible
            or not. We found that it gives low improvement into compression ratio.
            
    Result:
        list of dictionaries extracted from xml subtrees.
    '''
    df = []
    for child in listXML:
        row = {}
        row['tag'] = child.tag.replace('{'+namespace['mzXML']+'}', '')
        row['attrib'] = child.attrib
        grandchildren = child.getchildren()
        if len(grandchildren) == 0:
            row['children'] = {}
        else:
            row['children'] = parseTree(grandchildren, namespace)
        if row['tag'] == 'scan':
            mz, intensity = getMZIpair(child.text)
            row['data'] = {'mz': mz, 'intensity': intensity}
        elif row['tag'] == 'peaks':
            try:
                mz, intensity = getMZIpair(child.text)
                row['data'] = {'mz': mz, 'intensity': intensity}
            except:
                if use_numerize:
                    row['data'] = numerize(child.text)
                else:
                    row['data'] = child.text
        else:
            if use_numerize:
                row['data'] = numerize(child.text)
            else:
                row['data'] = child.text
        df.append(row)
    return df

def pickleSave(df, fname='data.pkl'):
    '''
    General routine for saving any data to be pickled with lzma comression.
    '''
    with open(fname, 'wb') as f:
        pickled = cPickle.dumps(df, protocol = -1)
        pickled = pylzma.compress( pickled,
                            dictionary=26,
                            fastBytes=255,
                            literalContextBits=3,
                            literalPosBits=0,
                            posBits=2,
                            algorithm=2,
                            eos=1,
                            multithreading=1
                                 )
        f.write(pickled)

def pickleLoad(fname):
    '''
    General routine for restoring pickled and lzma-compressed data.
    '''
    with open(fname, 'rb') as f:
        pickled = f.read()
        pickled = pylzma.decompress(pickled)
        df = cPickle.loads(pickled)
    return df

def saveBin(fname, savefname='data.pkl'):
    '''
    Convert mzXML data to pickled and compressed binary and save it as a file.
    '''
    df = mzXML2dict(fname)
    pickleSave(df, fname=savefname)
    
def saveMzXML(fname, savefname='data.mzXML'):
    '''
    Convert pickled and compressed binary back to mzXML file.
    '''
    df = pickleLoad(fname)
    fnm = savefname
    if not fnm.endswith('.mzXML'):
        fnm += '.mzXML'
    dict2mzXML(df, fnm)
    
def reconstructTree(treeElem, listDict):
    '''
    Subroutine to reconstruct xml tree structure from dictionarized version.
    '''
    for child in listDict:
        row = ET.SubElement(treeElem, child['tag'])
        row.attrib = child['attrib']
        grandchildren = child['children']
        
        if len(grandchildren) > 0:
            reconstructTree(row, grandchildren)
            
        if isinstance(child['data'], dict):
            mz = child['data']['mz']
            intensity = child['data']['intensity']
            text = setMZIpair(mz, intensity)
            row.text = text
        else:
            row.text = str(child['data'])
    return

def dict2mzXML(df, fname):
    '''
    Convert dictionarized mzXML file back to the original format.
    '''
    nm = df['namespace']
    nmK = nm.keys()[0]
    nmV = nm[nmK]
    root = ET.Element(nmK)
    root.attrib = df['attrib']
    root.attrib['xmlns'] = nmV
    children = df['children']
    if len(children) > 0:
        reconstructTree(root, children) 
    with open(fname, 'w') as f:
        xmlString = ET.tostring(root) # encoding='UTF-8', standalone="no"
        xmlMd = xmlmd.parseString(xmlString)
        xmlString = xmlMd.toprettyxml()
        xmlString = xmlString.replace(_HEAD_INCORRECT,
                                      _HEAD_CORRECT)
        f.write(xmlString)

def setMZIpair(mz, intensity):
    '''
    Convert numpy arrays with M/Z and intensity back to bytearray string.
    '''
    length = len(mz)
    assert length == len(intensity)
    size = 2*length
    toFormat = '>%dL' % size
    unString = []
    for ind in xrange(length):
        for fElement in [mz[ind], intensity[ind]]:
            iElement = struct.pack('f', float(fElement))
            Element = struct.unpack('I', iElement)
            unString.append(Element[0])
    procString = struct.pack(toFormat, *unString)
    string = base64.b64encode(procString)
    return string

def getFilenamesFromDir(dirname, extension):
    '''
    Get list with filenames of selected directory.
    '''
    fileList = []
    extVar = map(''.join, itertools.product(*zip(extension.upper(),
                                                 extension.lower())))
    extVar = tuple(extVar)
    for root, subfold, files in os.walk(dirname):
        for fnm in files:
            if fnm.endswith(extVar):
                fileList.append(root + '/' + fnm)
    return fileList

def multipleMzXML2Bin(dirname, rewrite=True):
    '''
    Multiple mzXML-binary conversion (for all files in specified directory).
    '''
    
    
        
    postfix = '_compressed'
    
    if dirname.endswith('/'):
        dirname = dirname[:-1]
    
    if not os.path.isdir(dirname+postfix):
        os.mkdir(dirname+postfix)
        
        
    
    extension = 'mzxml'
    fileList = getFilenamesFromDir(dirname, extension)
    i = 0
    N = len(fileList)
    for fnm in fileList:
        i += 1
        new_fnm = fnm.replace(dirname, dirname+postfix, 1)
        new_fnm = new_fnm[::-1].split('.', 1)[1]
        reqDirs = new_fnm.split('/', 1)[1]
        reqDirs = reqDirs[::-1]
        if not os.path.isdir(reqDirs):
            os.makedirs(reqDirs)
        new_fnm = new_fnm[::-1] + '.dat'
        try:
            if not rewrite:
                if os.path.isfile(new_fnm):
                    raise Exception('file %s is already exist' % new_fnm)
            saveBin(fnm, new_fnm)
            print '(%.2f %%) %s converted successfully' % (i*100./N, fnm)
        except Exception as exc:
            print exc.args
            print '(%.2f %%) %s did not converted' % (i*100./N, fnm)
        except:
            print '(%.2f %%) %s did not converted' % (i*100./N, fnm)

        
def multipleBin2MzXML(dirname, rewrite=True):
    '''
    Multiple binary-mzXML conversion (for all files in specified directory).
    '''
    postfix = '_decompressed'
    
    if dirname.endswith('/'):
        dirname = dirname[:-1]
    if not os.path.isdir(dirname+postfix):
        os.mkdir(dirname+postfix)
    extension = 'dat'
    fileList = getFilenamesFromDir(dirname, extension)
    i = 0
    N = len(fileList)
    for fnm in fileList:
        i += 1
        new_fnm = fnm.replace(dirname, dirname+postfix, 1)
        new_fnm = new_fnm[::-1].split('.', 1)[1]
        reqDirs = new_fnm.split('/', 1)[1]
        reqDirs = reqDirs[::-1]
        if not os.path.isdir(reqDirs):
            os.makedirs(reqDirs)
        new_fnm = new_fnm[::-1] + '.mzxml'
        try:
            if not rewrite:
                if os.path.isfile(new_fnm):
                    raise Exception('file %s is already exist' % new_fnm)
            saveMzXML(fnm, new_fnm)
            print '(%.2f %%) %s converted successfully' % (i*100./N, fnm)
        except Exception as exc:
            print exc.args
            print '(%.2f %%) %s did not converted' % (i*100./N, fnm)
        except:
            print '(%.2f %%) %s did not converted' % (i*100./N, fnm)
        

if __name__ == '__main__':
    dirname_mzxml = '../mzxml_data'
    dirname_compressed = '../mzxml_data_compressed'

    # uncomment to compress data from dirname_mzxml
    # multipleMzXML2Bin(dirname_mzxml)
    
    # uncomment to decompress data from dirname_compressed
    # multipleBin2MzXML(dirname_compressed)








#
