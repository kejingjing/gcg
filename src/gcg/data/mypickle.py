import pickle
import gzip

def dump(object, filename, protocol = 0):
    """Saves a compressed object to disk
    """
    file = gzip.GzipFile(filename, 'wb')
    file.write(pickle.dumps(object, protocol))
    file.close()

def load(filename):
    """Loads a compressed object from disk
    """
    file = gzip.GzipFile(filename, 'rb')
    buffer = b''
    while True:
        data = file.read()
        if data == b'':
            break
        buffer += data
    object = pickle.loads(buffer)
    file.close()
    return object
