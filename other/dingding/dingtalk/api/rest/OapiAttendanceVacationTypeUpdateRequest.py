'''
Created by auto_sdk on 2021.02.22
'''
from dingtalk.api.base import RestApi
class OapiAttendanceVacationTypeUpdateRequest(RestApi):
	def __init__(self,url=None):
		RestApi.__init__(self,url)
		self.biz_type = None
		self.extras = None
		self.hours_in_per_day = None
		self.leave_code = None
		self.leave_name = None
		self.leave_view_unit = None
		self.natural_day_leave = None
		self.op_userid = None
		self.paid_leave = None
		self.visibility_rules = None
		self.when_can_leave = None

	def getHttpMethod(self):
		return 'POST'

	def getapiname(self):
		return 'dingtalk.oapi.attendance.vacation.type.update'
