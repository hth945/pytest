from spyne import Application,rpc,ServiceBase,Iterable,Integer,Unicode
#如果支持soap的协议需要用到Soap11
from spyne.protocol.soap import Soap11
#如果开发传入传出为Json需要用到JsonDocument
from spyne.protocol.json import JsonDocument
#可以创建一个wsgi服务器，做测试用
from spyne.server.wsgi import WsgiApplication
#将Spyne创建的app 发布为django
# from django.views.decorators.csrf import csrf_exempt
#创建一个负责数据类型
from spyne.model.complex import ComplexModel
#引用其他的APP
# from ItemAPI import APP_Item



class HelloWorldService(ServiceBase):

    @rpc(Unicode, Integer, _returns=Iterable(Unicode))

    def say_hello(self, name, times):

        """Docstrings for service methods appear as documentation in the wsdl.

        <b>What fun!</b>

        @param name: the name to say hello to

        @param times: the number of times to say hello

        @return  When returning an iterable, you can use any type of python iterable. Here, we chose to use generators.

        """



        for i in range(times):

            yield u'Hello, %s' % name





# step2: Glue the service definition, input and output protocols

soap_app = Application([HelloWorldService], 'spyne.examples.hello.soap',

                       in_protocol=Soap11(validator='lxml'),

                       out_protocol=Soap11())



# step3: Wrap the Spyne application with its wsgi wrapper

wsgi_app = WsgiApplication(soap_app)



if __name__ == '__main__':

    import logging



    from wsgiref.simple_server import make_server



    # configure the python logger to show debugging output

    logging.basicConfig(level=logging.DEBUG)

    logging.getLogger('spyne.protocol.xml').setLevel(logging.DEBUG)



    logging.info("listening to http://127.0.0.1:8000")

    logging.info("wsdl is at: http://localhost:8000/?wsdl")



    # step4:Deploying the service using Soap via Wsgi

    # register the WSGI application as the handler to the wsgi server, and run the http server

    server = make_server('127.0.0.1', 8000, wsgi_app)

    server.serve_forever()


