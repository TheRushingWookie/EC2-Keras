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

ec2conn = boto.connect_ec2(
        aws_access_key_id = access_key,
        aws_secret_access_key = secret_key,
        )
cache = "./cache/"
#current = bucket.get_lifecycle_config()

def store_to_s3(file_name, bucket_name, contents):
    bucket = conn.get_bucket(bucket_name)
    k = Key(bucket)
    k.key = file_name
    k.set_contents_from_string(contents)

def get_from_s3(file_name, bucket_name):

    if file_name in os.listdir(cache):
        with open(cache + file_name) as file:
            return file.read()
    bucket = conn.get_bucket(bucket_name)
    k = Key(bucket)
    k.key = file_name
    string = StringIO.StringIO()
    k.get_contents_to_file(string)
    contents = string.getvalue()
    with open(cache + file_name, "w") as file:
        file.write(contents)
    return contents

def freeze_item():
    to_glacier = Transition(days=30, storage_class='GLACIER')
    rule = Rule('ruleid', '/', 'Enabled', transition=to_glacier)
    lifecycle = Lifecycle()
    lifecycle.append(rule)

def get_bucket_items(bucket_name):
    bucket = conn.get_bucket(bucket_name)
    return [ key.name.encode('utf-8') for key in bucket.list()]

def shutdown_spot_request():
    self_id = requests.get('http://instance-data/latest/meta-data/public-ipv4').text
    spot_reqs = ec2conn.get_all_spot_instance_requests()
    for spot_req in spot_reqs:
        if spot_req.instance_id = self_id:
            ec2conn.cancel_spot_instance_requests([spot_req.id,])
            ec2conn.stop_instances(instance_ids=[self_id,])
            return

