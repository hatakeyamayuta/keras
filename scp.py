import paramiko
with paramiko.SSHClient() as ssh:
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    ssh.connect('host',port=,username='',password='')
    stdin,stdout,stderr=ssh.exec_command('mkdir test')
