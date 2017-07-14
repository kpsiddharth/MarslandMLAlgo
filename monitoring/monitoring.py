from datetime import datetime
from elasticsearch_dsl import DocType, Date, Integer, Keyword, Text, Float, Long, Double
from elasticsearch_dsl.connections import connections

connections.create_connection(hosts=['localhost'])

class IterationError(DocType):
    iteration = Long()
    delta = Float()
    date = Long()
    
IterationError.init(index = 'ml_idx')