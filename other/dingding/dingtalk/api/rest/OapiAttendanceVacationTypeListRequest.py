'''
Created by auto_sdk on 2021.03.23
'''
from dingtalk.api.base import RestApi
class OapiAttendanceVacationTypeListRequest(RestApi):
	def __init__(self,url=None):
		RestApi.__init__(self,url)
		self.op_userid = None
		self.vacation_source = None

	def getHttpMethod(self):
		return 'POST'

	def getapiname(self):
		return 'dingtalk.oapi.attendance.vacation.type.list'
