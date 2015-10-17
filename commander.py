from flask.ext.api import FlaskAPI
import os
import boto
import boto.s3.connection
access_key = os.environ['AWS_ACCESS_KEY']
secret_key = os.environ['AWS_SECRET_KEY']

conn = boto.connect_s3(
        aws_access_key_id = access_key,
        aws_secret_access_key = secret_key,
        #is_secure=False,               # uncomment if you are not using ssl
        calling_format = boto.s3.connection.OrdinaryCallingFormat(),
        )


app = FlaskAPI(__name__)



@app.route('/get_job/<id>')
def example():
    return {'hello': 'world'}

app.run(host=os.environ['HOST'], post=int(os.environ['PORT']), debug=True)