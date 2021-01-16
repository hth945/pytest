from http.server import HTTPServer, BaseHTTPRequestHandler,SimpleHTTPRequestHandler
import json


# class Resquest(BaseHTTPRequestHandler):
class Resquest(SimpleHTTPRequestHandler):

    def __init__(self, *args,  **kwargs):
        super().__init__(*args, **kwargs)


    def handler(self):
        print("data:", self.rfile.readline().decode())
        self.wfile.write(self.rfile.readline())

    def do_GET(self):
        """Serve a GET request."""

        if self.path == '/hello':
            data = {
                'result_code': '1',
                'result_desc': 'Success',
                'timestamp': '',
                'data': {'message_id': '25d55ad283aa400af464c76d713c07ad'}
            }
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps(data).encode())
            return
        super().do_GET()


    def do_POST(self):
        print('self.headers-------------------------------------------------')
        print(self.headers)
        print('self.command-------------------------------------------------')
        print(self.command)

        print('self.req_datas.decode-------------------------------------------------')
        req_datas = self.rfile.read(int(self.headers['content-length']))  # 重点在此步!
        if self.path == '/runcmd':
            print('/runcmd:'+req_datas.decode())
        data = {
            'result_code': '2',
            'result_desc': 'Success',
            'timestamp': '',
            'data': {'message_id': '25d55ad283aa400af464c76d713c07ad'}
        }
        self.send_response(200)
        # self.send_header('Content-type', 'application/json')
        self.end_headers()
        self.wfile.write(json.dumps(data).encode('utf-8'))


if __name__ == '__main__':
    host = ('', 80)
    server = HTTPServer(host, Resquest)
    print("Starting server, listen at: %s:%s" % host)
    server.serve_forever()