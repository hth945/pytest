'''
Created by auto_sdk on 2021.02.04
'''
from dingtalk.api.base import RestApi
class OapiCustomerserviceMemberGetRequest(RestApi):
	def __init__(self,url=None):
		RestApi.__init__(self,url)
		self.production_type = None
		self.third_tenant_id = None
		self.user_id = None
		self.user_source_id = None

	def getHttpMethod(self):
		return 'POST'

	def getapiname(self):
		return 'dingtalk.oapi.customerservice.member.get'
