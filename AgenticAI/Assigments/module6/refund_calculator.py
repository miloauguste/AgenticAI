#!/usr/bin/env python3
"""
Refund Eligibility Calculator
Automated calculator for determining refund eligibility based on dates, thresholds, and policies
"""

import re
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
import logging

class RefundReason(Enum):
    """Enumeration of refund reasons"""
    DEFECTIVE = "defective"
    DAMAGED = "damaged"
    WRONG_ITEM = "wrong_item"
    NOT_AS_DESCRIBED = "not_as_described"
    CUSTOMER_CHANGED_MIND = "changed_mind"
    SHIPPING_DELAY = "shipping_delay"
    BILLING_ERROR = "billing_error"
    DUPLICATE_CHARGE = "duplicate_charge"
    UNAUTHORIZED_CHARGE = "unauthorized_charge"
    SERVICE_ISSUE = "service_issue"

class ProductType(Enum):
    """Enumeration of product types"""
    PHYSICAL = "physical"
    DIGITAL = "digital"
    SUBSCRIPTION = "subscription"
    SERVICE = "service"
    GIFT_CARD = "gift_card"

@dataclass
class RefundPolicy:
    """Represents a refund policy with conditions"""
    product_type: ProductType
    standard_days: int
    defective_days: Optional[int] = None
    conditions: List[str] = field(default_factory=list)
    exceptions: List[str] = field(default_factory=list)
    partial_refund_allowed: bool = True
    processing_time_days: int = 5

@dataclass
class Purchase:
    """Represents a customer purchase"""
    order_id: str
    purchase_date: datetime
    product_type: ProductType
    amount: float
    product_condition: str = "new"
    is_opened: bool = False
    has_original_packaging: bool = True
    is_activated: bool = False
    customer_type: str = "standard"

@dataclass
class RefundCalculation:
    """Result of refund eligibility calculation"""
    eligible: bool
    refund_amount: float
    processing_time_days: int
    reason: str
    conditions_met: List[str]
    conditions_failed: List[str]
    policy_applied: str
    calculation_details: Dict[str, Any]
    expiry_date: Optional[datetime] = None
    warnings: List[str] = field(default_factory=list)
    next_steps: List[str] = field(default_factory=list)

class RefundCalculator:
    """
    Intelligent refund eligibility calculator
    
    Features:
    - Date-based eligibility checking
    - Policy rule application
    - Partial refund calculations
    - Special condition handling
    - Multi-tier customer policies
    """
    
    def __init__(self):
        """Initialize refund calculator with default policies"""
        self.logger = logging.getLogger(__name__)
        self.policies = self._load_default_policies()
        self.special_conditions = self._load_special_conditions()
        
    def _load_default_policies(self) -> Dict[ProductType, RefundPolicy]:
        """Load default refund policies for different product types"""
        return {
            ProductType.PHYSICAL: RefundPolicy(
                product_type=ProductType.PHYSICAL,
                standard_days=30,
                defective_days=365,  # 1 year for defective items
                conditions=[
                    "Item must be unused",
                    "Original packaging required",
                    "All accessories included"
                ],
                exceptions=[
                    "Personalized items",
                    "Perishable goods",
                    "Undergarments"
                ],
                processing_time_days=5
            ),
            ProductType.DIGITAL: RefundPolicy(
                product_type=ProductType.DIGITAL,
                standard_days=14,
                defective_days=30,
                conditions=[
                    "Must not be downloaded/activated",
                    "License key unused"
                ],
                exceptions=[
                    "Software with security updates",
                    "One-time use licenses"
                ],
                processing_time_days=3
            ),
            ProductType.SUBSCRIPTION: RefundPolicy(
                product_type=ProductType.SUBSCRIPTION,
                standard_days=7,  # 7 days from billing
                defective_days=30,
                conditions=[
                    "Cancellation before next billing cycle",
                    "Minimal usage during period"
                ],
                exceptions=[
                    "Annual subscriptions (pro-rated)",
                    "Promotional subscriptions"
                ],
                partial_refund_allowed=True,
                processing_time_days=7
            ),
            ProductType.SERVICE: RefundPolicy(
                product_type=ProductType.SERVICE,
                standard_days=14,
                defective_days=60,
                conditions=[
                    "Service not yet performed",
                    "At least 24 hours notice"
                ],
                exceptions=[
                    "Emergency services",
                    "Custom services already started"
                ],
                processing_time_days=3
            ),
            ProductType.GIFT_CARD: RefundPolicy(
                product_type=ProductType.GIFT_CARD,
                standard_days=0,  # Generally no refunds
                defective_days=30,  # Only for technical issues
                conditions=[
                    "Technical malfunction only",
                    "Card not activated/used"
                ],
                exceptions=[
                    "Promotional gift cards",
                    "Bonus gift cards"
                ],
                partial_refund_allowed=False,
                processing_time_days=5
            )
        }
    
    def _load_special_conditions(self) -> Dict[str, Dict[str, Any]]:
        """Load special conditions and modifiers"""
        return {
            "premium_customer": {
                "additional_days": 15,
                "expedited_processing": True,
                "partial_refund_boost": 0.1
            },
            "holiday_season": {
                "extended_deadline": "2024-01-31",  # Extended return until end of January
                "applicable_period": ("2023-11-01", "2023-12-31")
            },
            "defective_batch": {
                "product_ids": ["BATCH_2023_A", "BATCH_2023_B"],
                "full_refund_guaranteed": True,
                "expedited_processing": True
            },
            "shipping_delay": {
                "threshold_days": 10,  # If shipping delayed by 10+ days
                "compensation_percentage": 20,  # 20% compensation
                "full_refund_option": True
            }
        }
    
    def calculate_refund_eligibility(self, purchase: Purchase, refund_reason: RefundReason,
                                   request_date: datetime = None) -> RefundCalculation:
        """
        Calculate refund eligibility and amount
        
        Args:
            purchase: Purchase details
            refund_reason: Reason for refund request
            request_date: Date of refund request (default: now)
            
        Returns:
            RefundCalculation with eligibility and details
        """
        if request_date is None:
            request_date = datetime.now()
        
        try:
            # Get applicable policy
            policy = self.policies.get(purchase.product_type)
            if not policy:
                return self._create_error_result("No policy found for product type")
            
            # Calculate base eligibility
            base_result = self._calculate_base_eligibility(purchase, refund_reason, policy, request_date)
            
            # Apply special conditions
            enhanced_result = self._apply_special_conditions(base_result, purchase, refund_reason, request_date)
            
            # Calculate refund amount
            final_result = self._calculate_refund_amount(enhanced_result, purchase, refund_reason, policy)
            
            # Add next steps and warnings
            final_result = self._add_guidance(final_result, purchase, refund_reason)
            
            self.logger.info(f"Calculated refund eligibility for order {purchase.order_id}: {final_result.eligible}")
            return final_result
            
        except Exception as e:
            self.logger.error(f"Error calculating refund eligibility: {str(e)}")
            return self._create_error_result(f"Calculation error: {str(e)}")
    
    def bulk_calculate_eligibility(self, purchases: List[Tuple[Purchase, RefundReason]]) -> List[RefundCalculation]:
        """
        Calculate refund eligibility for multiple purchases
        
        Args:
            purchases: List of (Purchase, RefundReason) tuples
            
        Returns:
            List of RefundCalculation results
        """
        results = []
        for purchase, reason in purchases:
            result = self.calculate_refund_eligibility(purchase, reason)
            results.append(result)
        
        return results
    
    def estimate_processing_time(self, purchase: Purchase, refund_reason: RefundReason) -> Dict[str, Any]:
        """
        Estimate refund processing time
        
        Args:
            purchase: Purchase details
            refund_reason: Reason for refund
            
        Returns:
            Processing time estimates
        """
        policy = self.policies.get(purchase.product_type)
        base_days = policy.processing_time_days if policy else 5
        
        # Adjust for special conditions
        if purchase.customer_type == "premium":
            base_days = max(1, base_days - 2)
        
        if refund_reason in [RefundReason.DEFECTIVE, RefundReason.BILLING_ERROR]:
            base_days = max(1, base_days - 1)  # Expedited for these reasons
        
        return {
            "estimated_days": base_days,
            "business_days_only": True,
            "expedited_available": purchase.customer_type == "premium",
            "estimated_completion": datetime.now() + timedelta(days=base_days),
            "factors": self._get_processing_factors(purchase, refund_reason)
        }
    
    def validate_refund_request(self, purchase: Purchase, refund_reason: RefundReason,
                              additional_info: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Validate a refund request for completeness
        
        Args:
            purchase: Purchase details
            refund_reason: Reason for refund
            additional_info: Additional information provided
            
        Returns:
            Validation results and missing information
        """
        validation = {
            "is_valid": True,
            "missing_info": [],
            "recommendations": [],
            "additional_docs_needed": []
        }
        
        # Check required information based on reason
        if refund_reason in [RefundReason.DEFECTIVE, RefundReason.DAMAGED]:
            if not additional_info or "photos" not in additional_info:
                validation["missing_info"].append("Photos of defective/damaged item")
            if not additional_info or "description" not in additional_info:
                validation["missing_info"].append("Description of the defect/damage")
        
        if refund_reason == RefundReason.WRONG_ITEM:
            if not additional_info or "received_item" not in additional_info:
                validation["missing_info"].append("Description of item received")
            if not additional_info or "expected_item" not in additional_info:
                validation["missing_info"].append("Description of expected item")
        
        if refund_reason == RefundReason.SHIPPING_DELAY:
            if not additional_info or "expected_delivery" not in additional_info:
                validation["missing_info"].append("Expected delivery date")
            if not additional_info or "actual_delivery" not in additional_info:
                validation["missing_info"].append("Actual delivery date (if delivered)")
        
        # Check for additional documentation
        if purchase.amount > 500:
            validation["additional_docs_needed"].append("Purchase receipt/invoice")
        
        if purchase.product_type == ProductType.DIGITAL and purchase.is_activated:
            validation["additional_docs_needed"].append("Proof of technical issue")
        
        # Generate recommendations
        if validation["missing_info"]:
            validation["is_valid"] = False
            validation["recommendations"].append("Gather missing information before processing")
        
        if len(validation["additional_docs_needed"]) > 0:
            validation["recommendations"].append("Request additional documentation")
        
        return validation
    
    def get_policy_details(self, product_type: ProductType) -> Dict[str, Any]:
        """Get detailed policy information for a product type"""
        policy = self.policies.get(product_type)
        if not policy:
            return {"error": "Policy not found"}
        
        return {
            "product_type": product_type.value,
            "standard_return_period": f"{policy.standard_days} days",
            "defective_return_period": f"{policy.defective_days} days" if policy.defective_days else "Same as standard",
            "conditions": policy.conditions,
            "exceptions": policy.exceptions,
            "partial_refunds_allowed": policy.partial_refund_allowed,
            "processing_time": f"{policy.processing_time_days} business days",
            "special_notes": self._get_policy_notes(product_type)
        }
    
    # Private helper methods
    
    def _calculate_base_eligibility(self, purchase: Purchase, reason: RefundReason,
                                  policy: RefundPolicy, request_date: datetime) -> RefundCalculation:
        """Calculate base eligibility before special conditions"""
        days_since_purchase = (request_date - purchase.purchase_date).days
        conditions_met = []
        conditions_failed = []
        eligible = True
        
        # Check time eligibility
        if reason in [RefundReason.DEFECTIVE, RefundReason.DAMAGED]:
            time_limit = policy.defective_days or policy.standard_days
            policy_applied = "Defective Item Policy"
        else:
            time_limit = policy.standard_days
            policy_applied = "Standard Return Policy"
        
        if days_since_purchase <= time_limit:
            conditions_met.append(f"Within {time_limit} day time limit")
        else:
            conditions_failed.append(f"Exceeds {time_limit} day time limit")
            eligible = False
        
        # Check product-specific conditions
        if purchase.product_type == ProductType.PHYSICAL:
            if not purchase.is_opened and purchase.has_original_packaging:
                conditions_met.append("Item unopened with original packaging")
            elif purchase.is_opened:
                if reason in [RefundReason.DEFECTIVE, RefundReason.DAMAGED, RefundReason.WRONG_ITEM]:
                    conditions_met.append("Opened item acceptable for defective/wrong item")
                else:
                    conditions_failed.append("Item has been opened")
                    eligible = False
        
        elif purchase.product_type == ProductType.DIGITAL:
            if not purchase.is_activated:
                conditions_met.append("Digital product not activated")
            else:
                if reason in [RefundReason.DEFECTIVE, RefundReason.SERVICE_ISSUE]:
                    conditions_met.append("Activated product acceptable for technical issues")
                else:
                    conditions_failed.append("Digital product already activated")
                    eligible = False
        
        # Calculate expiry date
        expiry_date = purchase.purchase_date + timedelta(days=time_limit)
        
        return RefundCalculation(
            eligible=eligible,
            refund_amount=purchase.amount if eligible else 0.0,
            processing_time_days=policy.processing_time_days,
            reason=f"Base eligibility check: {'PASSED' if eligible else 'FAILED'}",
            conditions_met=conditions_met,
            conditions_failed=conditions_failed,
            policy_applied=policy_applied,
            calculation_details={
                "days_since_purchase": days_since_purchase,
                "time_limit": time_limit,
                "refund_reason": reason.value
            },
            expiry_date=expiry_date
        )
    
    def _apply_special_conditions(self, base_result: RefundCalculation, purchase: Purchase,
                                reason: RefundReason, request_date: datetime) -> RefundCalculation:
        """Apply special conditions and exceptions"""
        # Premium customer benefits
        if purchase.customer_type == "premium":
            special_conditions = self.special_conditions["premium_customer"]
            
            if not base_result.eligible:
                # Check if premium customer gets extended time
                extended_days = base_result.calculation_details["time_limit"] + special_conditions["additional_days"]
                days_since_purchase = base_result.calculation_details["days_since_purchase"]
                
                if days_since_purchase <= extended_days:
                    base_result.eligible = True
                    base_result.conditions_met.append("Premium customer extended return period")
                    base_result.reason = "Eligible under premium customer policy"
                    base_result.refund_amount = purchase.amount
            
            if special_conditions.get("expedited_processing"):
                base_result.processing_time_days = max(1, base_result.processing_time_days - 2)
                base_result.conditions_met.append("Expedited processing for premium customer")
        
        # Holiday season extensions
        holiday_conditions = self.special_conditions["holiday_season"]
        purchase_date_str = purchase.purchase_date.strftime("%Y-%m-%d")
        holiday_start, holiday_end = holiday_conditions["applicable_period"]
        
        if holiday_start <= purchase_date_str <= holiday_end:
            holiday_deadline = datetime.strptime(holiday_conditions["extended_deadline"], "%Y-%m-%d")
            if request_date <= holiday_deadline:
                if not base_result.eligible:
                    base_result.eligible = True
                    base_result.refund_amount = purchase.amount
                base_result.conditions_met.append("Holiday season extended return period")
                base_result.reason = "Eligible under holiday return policy"
        
        # Shipping delay compensation
        if reason == RefundReason.SHIPPING_DELAY:
            shipping_conditions = self.special_conditions["shipping_delay"]
            # This would normally check actual shipping data
            # For demo, assume delay qualifies for compensation
            base_result.eligible = True
            base_result.conditions_met.append("Shipping delay compensation available")
            base_result.warnings.append("Additional compensation may apply for shipping delays")
        
        return base_result
    
    def _calculate_refund_amount(self, result: RefundCalculation, purchase: Purchase,
                               reason: RefundReason, policy: RefundPolicy) -> RefundCalculation:
        """Calculate final refund amount"""
        if not result.eligible:
            result.refund_amount = 0.0
            return result
        
        base_amount = purchase.amount
        
        # Full refund cases
        if reason in [RefundReason.DEFECTIVE, RefundReason.DAMAGED, RefundReason.WRONG_ITEM, 
                     RefundReason.BILLING_ERROR, RefundReason.DUPLICATE_CHARGE, RefundReason.UNAUTHORIZED_CHARGE]:
            result.refund_amount = base_amount
            result.calculation_details["refund_type"] = "full"
        
        # Partial refund cases
        elif reason == RefundReason.CUSTOMER_CHANGED_MIND:
            if purchase.product_type == ProductType.SUBSCRIPTION:
                # Pro-rated refund for subscriptions
                days_used = (datetime.now() - purchase.purchase_date).days
                billing_cycle_days = 30  # Assume monthly billing
                unused_ratio = max(0, (billing_cycle_days - days_used) / billing_cycle_days)
                result.refund_amount = base_amount * unused_ratio
                result.calculation_details["refund_type"] = "prorated"
                result.calculation_details["unused_ratio"] = unused_ratio
            else:
                # Standard refund with possible restocking fee
                restocking_fee = 0.15 if purchase.is_opened else 0.0
                result.refund_amount = base_amount * (1 - restocking_fee)
                result.calculation_details["refund_type"] = "standard_with_fee"
                result.calculation_details["restocking_fee"] = restocking_fee
        
        # Service issues
        elif reason == RefundReason.SERVICE_ISSUE:
            # Depending on severity, could be partial or full
            result.refund_amount = base_amount * 0.8  # 80% refund for service issues
            result.calculation_details["refund_type"] = "service_adjustment"
        
        # Shipping delay compensation
        elif reason == RefundReason.SHIPPING_DELAY:
            compensation_rate = self.special_conditions["shipping_delay"]["compensation_percentage"] / 100
            result.refund_amount = min(base_amount * compensation_rate, base_amount)
            result.calculation_details["refund_type"] = "shipping_compensation"
        
        # Round to 2 decimal places
        result.refund_amount = round(result.refund_amount, 2)
        
        return result
    
    def _add_guidance(self, result: RefundCalculation, purchase: Purchase, reason: RefundReason) -> RefundCalculation:
        """Add next steps and warnings"""
        # Next steps
        if result.eligible:
            result.next_steps.append("Process refund request")
            result.next_steps.append("Send confirmation email to customer")
            
            if reason in [RefundReason.DEFECTIVE, RefundReason.DAMAGED]:
                result.next_steps.append("Arrange product return shipping")
                result.next_steps.append("Inspect returned item")
            
            if purchase.amount > 500:
                result.next_steps.append("Manager approval required for high-value refund")
                result.warnings.append("High-value refund requires additional approval")
        else:
            result.next_steps.append("Explain policy limitations to customer")
            result.next_steps.append("Offer alternative solutions if available")
            result.next_steps.append("Document refund denial reason")
        
        # Warnings
        if purchase.product_type == ProductType.DIGITAL and result.eligible:
            result.warnings.append("Verify technical issue before processing digital refund")
        
        if reason == RefundReason.CUSTOMER_CHANGED_MIND and result.refund_amount < purchase.amount:
            result.warnings.append("Customer should be informed of partial refund amount")
        
        return result
    
    def _create_error_result(self, error_message: str) -> RefundCalculation:
        """Create error result for failed calculations"""
        return RefundCalculation(
            eligible=False,
            refund_amount=0.0,
            processing_time_days=0,
            reason=f"Error: {error_message}",
            conditions_met=[],
            conditions_failed=[error_message],
            policy_applied="Error",
            calculation_details={"error": True},
            warnings=[error_message]
        )
    
    def _get_processing_factors(self, purchase: Purchase, reason: RefundReason) -> List[str]:
        """Get factors affecting processing time"""
        factors = []
        
        if purchase.amount > 1000:
            factors.append("High value requires additional verification")
        
        if reason in [RefundReason.DEFECTIVE, RefundReason.DAMAGED]:
            factors.append("Physical inspection required")
        
        if purchase.customer_type == "premium":
            factors.append("Premium customer - expedited processing")
        
        if purchase.product_type == ProductType.DIGITAL:
            factors.append("Digital product - faster processing")
        
        return factors
    
    def _get_policy_notes(self, product_type: ProductType) -> List[str]:
        """Get additional policy notes for product type"""
        notes = {
            ProductType.PHYSICAL: [
                "Items must be in resellable condition",
                "Original receipt may be required for high-value items",
                "Shipping costs are non-refundable unless item is defective"
            ],
            ProductType.DIGITAL: [
                "Refunds only available if not downloaded/activated",
                "Technical issues may qualify for exception",
                "License keys must be unused"
            ],
            ProductType.SUBSCRIPTION: [
                "Cancellation takes effect at end of billing period",
                "Pro-rated refunds available in some cases",
                "Auto-renewal can be disabled separately"
            ],
            ProductType.SERVICE: [
                "24-hour cancellation notice required",
                "Custom services may have different terms",
                "Partial completion may affect refund amount"
            ],
            ProductType.GIFT_CARD: [
                "Generally non-refundable",
                "Technical issues may qualify for replacement",
                "Check local laws for specific requirements"
            ]
        }
        
        return notes.get(product_type, [])


def main():
    """Demonstration of refund calculator"""
    print("Refund Eligibility Calculator Demo")
    print("=" * 50)
    
    # Initialize calculator
    calculator = RefundCalculator()
    
    # Test cases
    test_purchases = [
        Purchase(
            order_id="ORD-001",
            purchase_date=datetime.now() - timedelta(days=20),
            product_type=ProductType.PHYSICAL,
            amount=299.99,
            is_opened=False,
            has_original_packaging=True
        ),
        Purchase(
            order_id="ORD-002", 
            purchase_date=datetime.now() - timedelta(days=45),
            product_type=ProductType.DIGITAL,
            amount=49.99,
            is_activated=True
        ),
        Purchase(
            order_id="ORD-003",
            purchase_date=datetime.now() - timedelta(days=10),
            product_type=ProductType.SUBSCRIPTION,
            amount=99.99,
            customer_type="premium"
        )
    ]
    
    test_reasons = [
        RefundReason.CUSTOMER_CHANGED_MIND,
        RefundReason.DEFECTIVE,
        RefundReason.CUSTOMER_CHANGED_MIND
    ]
    
    print("Testing refund calculations:")
    print("-" * 30)
    
    for purchase, reason in zip(test_purchases, test_reasons):
        print(f"\nOrder: {purchase.order_id}")
        print(f"Product: {purchase.product_type.value}")
        print(f"Amount: ${purchase.amount}")
        print(f"Reason: {reason.value}")
        
        result = calculator.calculate_refund_eligibility(purchase, reason)
        
        print(f"Eligible: {result.eligible}")
        print(f"Refund Amount: ${result.refund_amount}")
        print(f"Processing Time: {result.processing_time_days} days")
        print(f"Policy: {result.policy_applied}")
        
        if result.conditions_met:
            print(f"Conditions Met: {', '.join(result.conditions_met[:2])}")
        
        if result.conditions_failed:
            print(f"Conditions Failed: {', '.join(result.conditions_failed[:2])}")
        
        print("-" * 30)
    
    # Show policy details
    print("\nPolicy Summary:")
    for product_type in ProductType:
        details = calculator.get_policy_details(product_type)
        if "error" not in details:
            print(f"{product_type.value}: {details['standard_return_period']}")
    
    print("\nRefund calculator demonstration complete!")


if __name__ == "__main__":
    main()