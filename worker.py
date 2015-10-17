import boto
import boto.s3.connection
from boto.s3.key import Key
from boto.s3.lifecycle import Lifecycle, Transition, Rule
import os
import StringIO

access_key = os.environ['AWS_ACCESS_KEY']
secret_key = os.environ['AWS_SECRET_KEY']

conn = boto.connect_s3(
        aws_access_key_id = access_key,
        aws_secret_access_key = secret_key,
        )
bucket = conn.get_bucket('kerasmodels')
#current = bucket.get_lifecycle_config()

def store_to_s3(file_name, contents):
    k = Key(bucket)
    k.key = file_name
    k.set_contents_from_string(contents)

def get_from_s3(file_name):
    k = Key(bucket)
    k.key = file_name
    string = StringIO.StringIO()
    k.get_contents_to_file(string)
    return string.getvalue()

def freeze_item():
    to_glacier = Transition(days=30, storage_class='GLACIER')
    rule = Rule('ruleid', '/', 'Enabled', transition=to_glacier)
    lifecycle = Lifecycle()
    lifecycle.append(rule)

def get_bucket_items():
    return [ key.name.encode('utf-8') for key in bucket.list()]