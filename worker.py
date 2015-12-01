import boto
import boto.s3.connection
from boto.s3.key import Key
from boto.s3.lifecycle import Lifecycle, Transition, Rule

import os
import StringIO
import requests
debug_mode = os.environ.get('LOCALDEBUG')
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

def get_spot_instance_request(instance_id, any_req=False):
    spot_reqs = ec2conn.get_all_spot_instance_requests()
    if any_req:
        return spot_reqs[0]
    for spot_req in spot_reqs:
        print spot_req.instance_id, self_id
        if spot_req.instance_id == instance_id:
            return spot_req

def shutdown_spot_request():
    self_id = requests.get('http://instance-data/latest/meta-data/instance-id').text
    request = get_spot_instance_request_id(self_id)
    ec2conn.cancel_spot_instance_requests([request.id,])
    ec2conn.terminate_instances(instance_ids=[self_id,])

shutdown_data = {"to" : "quinnjarr@gmail.com",
              "from" : "quinnjarr@gmail.com",
              "subject" : "Startup",
              "html" : "Starting ec2 cats vs dogs",
              }
shutdown_data['html'] = "Shutting down cats vs dogs"
shutdown_data['subject'] = "Early shutdown" 
def check_for_early_shutdown():
    self_id = requests.get('http://instance-data/latest/meta-data/instance-id').text if not debug_mode else None
    spot_req = get_spot_instance_request(self_id, any_req=debug_mode)
    if spot_req.status.code == u'marked-for-termination':
        return True
    return False
