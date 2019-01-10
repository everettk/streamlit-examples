#!/bin/sh -e

function usage() {
  cat <<EOF
Prerequisites.
* Atom installed
* sshfs installed
* EC2 external IP address is known ie cloudformation template has been run

$0 '<AWS EC2 External IP Address>' '/path/to/ssh_pem_file'

For example:
$0 35.166.122.127 ~/.ssh/streamlit.pem
EOF
  exit 1
}

function create_ssh_config() {
  chmod 600 ${KEY}
  touch ~/.ssh/config
  cat <<EOF >> ~/.ssh/config

Host streamlit-aws
  Hostname ${IP}
  User ubuntu
  IdentityFile ${KEY}

EOF
}

function install_streamlit_atom() {
  apm list --installed --bare | grep -q streamlit-atom || apm install streamlit-atom
}

function configure_streamlit_atom() {
  # Not ideal but it works right now.
  sed -i -e "s/localhost:8501/${IP}:8501/g" ~/.atom/packages/streamlit-atom/lib/streamlit-atom.js
}

function next_steps() {
  cat <<EOF
Next steps:

sshfs streamlit-aws:src ~/remote-src

atom ~/remote-src/verify.py
Save file.

ssh streamlit-aws
python ~/remote-src/verify.py

In Atom, Ctrl-Alt-O
EOF
}

IP=$1
KEY=$2

if [ -z $IP -o -z $KEY ] ; then
  usage
fi

mkdir -p ~/.ssh ~/remote-src

install_streamlit_atom
configure_streamlit_atom

grep -q ${IP} ~/.ssh/config || create_ssh_config

next_steps
