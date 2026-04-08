from __future__ import annotations

from dataclasses import dataclass

from .models import ProfessionProfile, ProfessionRow
from .text_utils import unique_preserve_order


@dataclass(slots=True)
class DomainRule:
    domain: str
    keywords: tuple[str, ...]
    authority_level: str
    resource_access: tuple[str, ...]
    target_population: tuple[str, ...]
    workplace_context: tuple[str, ...]
    social_sensitivity: tuple[str, ...]
    victimization_surface: tuple[str, ...]
    profile_tags: tuple[str, ...]


DOMAIN_RULES: tuple[DomainRule, ...] = (
    DomainRule(
        domain="public_sector",
        keywords=("委员会", "党组织", "机关", "法院", "检察院", "政协", "工会", "妇女联合会", "负责人"),
        authority_level="very_high",
        resource_access=("公共权力流程", "组织内部通道", "敏感内部信息"),
        target_population=("公众", "被管理对象", "下属"),
        workplace_context=("层级化组织", "制度审批流程", "跨部门协调"),
        social_sensitivity=("公权力", "程序正义", "公共舆论"),
        victimization_surface=("问责压力", "舆论污名", "层级压制"),
        profile_tags=("公权力", "审批", "组织管理"),
    ),
    DomainRule(
        domain="education",
        keywords=("学校", "教育", "校长"),
        authority_level="high",
        resource_access=("学生信息", "评价与分配权", "招生升学资源"),
        target_population=("学生", "家长", "教师"),
        workplace_context=("校园场景", "教学评价", "未成年人保护"),
        social_sensitivity=("未成年人", "教育公平", "师德舆情"),
        victimization_surface=("家校冲突", "绩效压力", "舆论道德审判"),
        profile_tags=("教育", "未成年人", "评价权"),
    ),
    DomainRule(
        domain="healthcare",
        keywords=("卫生", "医疗", "医学"),
        authority_level="high",
        resource_access=("患者敏感信息", "诊疗资源", "药品与流程权限"),
        target_population=("患者", "家属", "同事"),
        workplace_context=("医疗机构", "高压服务场景", "急慢性诊疗流程"),
        social_sensitivity=("生命健康", "隐私", "医疗纠纷"),
        victimization_surface=("医患冲突", "高强度轮班", "暴力威胁"),
        profile_tags=("医疗", "患者", "敏感信息"),
    ),
    DomainRule(
        domain="research",
        keywords=("研究人员", "科研"),
        authority_level="medium",
        resource_access=("科研经费", "实验数据", "知识产权"),
        target_population=("学生助手", "同行", "课题合作方"),
        workplace_context=("实验室", "项目申报", "学术评价"),
        social_sensitivity=("学术诚信", "知识产权", "科研伦理"),
        victimization_surface=("成果剥夺", "项目压榨", "评价歧视"),
        profile_tags=("科研", "实验", "数据"),
    ),
    DomainRule(
        domain="enterprise_management",
        keywords=("董事", "经理", "主管", "企业负责人"),
        authority_level="very_high",
        resource_access=("预算与审批", "用工分配权", "商业机密"),
        target_population=("员工", "客户", "供应商"),
        workplace_context=("业绩压力", "用工关系", "供应链协作"),
        social_sensitivity=("雇佣公平", "商业合规", "劳动关系"),
        victimization_surface=("绩效淘汰", "高压文化", "利益冲突"),
        profile_tags=("管理", "预算", "用工"),
    ),
    DomainRule(
        domain="engineering",
        keywords=("工程技术人员", "工程师", "勘探", "设计", "制造", "设备", "测量", "火药", "车辆", "火控", "飞机"),
        authority_level="medium",
        resource_access=("设备系统", "工艺流程", "安全规范"),
        target_population=("工友", "客户", "现场作业人员"),
        workplace_context=("生产现场", "项目交付", "安全检查"),
        social_sensitivity=("安全生产", "质量责任", "技术保密"),
        victimization_surface=("工伤风险", "超时施工", "责任甩锅"),
        profile_tags=("工程", "现场", "安全"),
    ),
    DomainRule(
        domain="military",
        keywords=("军人", "兵器"),
        authority_level="high",
        resource_access=("武器装备", "纪律体系", "训练资源"),
        target_population=("下属", "协作人员", "公众"),
        workplace_context=("纪律化体系", "训练任务", "保密场景"),
        social_sensitivity=("国家安全", "纪律约束", "高风险装备"),
        victimization_surface=("高压指令", "危险任务", "创伤压力"),
        profile_tags=("纪律", "装备", "高风险"),
    ),
)

DOMAIN_LABELS = {
    "public_sector": "公共部门",
    "education": "教育系统",
    "healthcare": "医疗卫生",
    "research": "科研系统",
    "enterprise_management": "企业管理",
    "engineering": "工程技术",
    "military": "军警/高风险纪律体系",
    "general_workforce": "通用职场",
}


class ProfessionProfileAgent:
    def build_profession_profile(self, profession: ProfessionRow) -> ProfessionProfile:
        name = profession.profession_name_norm
        matched_rules = [rule for rule in DOMAIN_RULES if any(keyword in name for keyword in rule.keywords)]
        if not matched_rules:
            matched_rules = [
                DomainRule(
                    domain="general_workforce",
                    keywords=(),
                    authority_level="medium",
                    resource_access=("岗位资源", "业务流程", "内部信息"),
                    target_population=("同事", "客户", "合作对象"),
                    workplace_context=("岗位协作", "绩效压力", "制度流程"),
                    social_sensitivity=("职业评价", "劳动关系", "内部层级"),
                    victimization_surface=("超时劳动", "羞辱性管理", "机会剥夺"),
                    profile_tags=("通用职业", "流程", "协作"),
                )
            ]

        authority_priority = {"frontline": 0, "medium": 1, "high": 2, "very_high": 3}
        authority_level = max(
            (rule.authority_level for rule in matched_rules),
            key=lambda level: authority_priority.get(level, 0),
        )
        resource_access = unique_preserve_order(item for rule in matched_rules for item in rule.resource_access)
        target_population = unique_preserve_order(item for rule in matched_rules for item in rule.target_population)
        workplace_context = unique_preserve_order(item for rule in matched_rules for item in rule.workplace_context)
        social_sensitivity = unique_preserve_order(item for rule in matched_rules for item in rule.social_sensitivity)
        victimization_surface = unique_preserve_order(item for rule in matched_rules for item in rule.victimization_surface)
        profile_tags = unique_preserve_order(
            [DOMAIN_LABELS.get(rule.domain, rule.domain) for rule in matched_rules]
            + [item for rule in matched_rules for item in rule.profile_tags]
        )
        domain_label = DOMAIN_LABELS.get(matched_rules[0].domain, matched_rules[0].domain)
        profile_summary = (
            f"{name}通常处于{domain_label}领域，"
            f"拥有{resource_access[0]}等岗位资源，"
            f"经常接触{target_population[0]}，"
            f"既可能涉及{social_sensitivity[0]}，也容易遭遇{victimization_surface[0]}。"
        )

        return ProfessionProfile(
            profession_id=profession.profession_id,
            profession_name=name,
            domain=matched_rules[0].domain,
            authority_level=authority_level,
            resource_access=resource_access,
            target_population=target_population,
            workplace_context=workplace_context,
            social_sensitivity=social_sensitivity,
            victimization_surface=victimization_surface,
            profile_tags=profile_tags,
            profile_summary=profile_summary,
        )
