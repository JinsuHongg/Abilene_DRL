from mininet.net import Mininet
from mininet.node import Controller
from mininet.cli import CLI

net = Mininet(controller=Controller)
net.addController('c0', controller=Controller, command='ovs-controller')

c0 = net.addController('c0')
s1 = net.addSwitch('s1')
s2 = net.addSwitch('s2')
net.addLink(s1, s2)

net.start()
CLI(net)
net.stop()