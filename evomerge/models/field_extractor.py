"""
Field extraction from OCR results.

This module implements field extraction from OCR results for Japanese invoices and receipts.
"""

import re
import logging
from typing import Dict, Any, List, Tuple, Optional, Union, Pattern
import numpy as np

# Set up logger
logger = logging.getLogger(__name__)


class InvoiceFieldExtractor:
    """
    Extract structured fields from Japanese invoice and receipt OCR results.
    
    This class implements field extraction logic for common fields in Japanese
    invoices and receipts, such as dates, amounts, tax information, etc.
    """
    
    def __init__(
        self,
        confidence_threshold: float = 0.7,
        use_regex: bool = True,
        use_layout_analysis: bool = True,
        config: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize the field extractor.
        
        Args:
            confidence_threshold: Minimum confidence threshold for field extraction
            use_regex: Whether to use regex patterns for field extraction
            use_layout_analysis: Whether to use layout analysis for field extraction
            config: Additional configuration
        """
        self.confidence_threshold = confidence_threshold
        self.use_regex = use_regex
        self.use_layout_analysis = use_layout_analysis
        self.config = config or {}
        
        # Initialize regex patterns for Japanese invoices/receipts
        self._init_regex_patterns()
        
        # Field type mappings
        self.field_types = {
            "invoice_number": str,
            "receipt_number": str,
            "date": str,
            "due_date": str,
            "subtotal": str,
            "tax": str,
            "total_amount": str,
            "company_name": str,
            "postal_code": str,
            "address": str,
            "phone": str,
            "email": str,
        }
    
    def _init_regex_patterns(self) -> None:
        """Initialize regex patterns for field extraction."""
        # Date patterns (Japanese and Western formats)
        self.date_patterns = [
            # Japanese format: 令和5年6月25日, 2023年6月25日, etc.
            r'(令和|平成|昭和)?(\d{1,4})年(\d{1,2})月(\d{1,2})日',
            # Western format: 2023/06/25, 2023-06-25, etc.
            r'(\d{4})[-/](\d{1,2})[-/](\d{1,2})',
            # Short Japanese format: 6月25日
            r'(\d{1,2})月(\d{1,2})日',
        ]
        
        # Amount patterns
        self.amount_patterns = [
            # With currency symbol, with/without comma
            r'¥\s?([\d,]+)',
            r'(\d+[,\d]*)\s?円',
            # Tax patterns
            r'消費税\s?([\d,]+)円?',
            r'税額\s?([\d,]+)円?',
            r'税\(?(\d+)%\)?\s*([\d,]+)円?',
        ]
        
        # Invoice/receipt number patterns
        self.number_patterns = [
            r'請求書番号[:\s]*([\w\d\-]+)',
            r'納品書番号[:\s]*([\w\d\-]+)',
            r'領収書番号[:\s]*([\w\d\-]+)',
            r'No\.?\s*([\w\d\-]+)',
            r'#\s*([\w\d\-]+)',
        ]
        
        # Postal code patterns
        self.postal_patterns = [
            r'〒\s*(\d{3}[-－]\d{4})',
            r'郵便番号\s*[:\s]*(\d{3}[-－]\d{4})',
        ]
        
        # Phone patterns
        self.phone_patterns = [
            r'電話[番号]?[:\s]*([\d\-（）\(\)]+)',
            r'TEL[:\s]*([\d\-（）\(\)]+)',
            r'Tel[:\s]*([\d\-（）\(\)]+)',
        ]
        
        # Email patterns
        self.email_patterns = [
            r'[Ee][-]?[Mm]ail[:\s]*([a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,})',
            r'メール[:\s]*([a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,})',
        ]
        
        # Company name patterns
        self.company_patterns = [
            r'(株式会社|合同会社|有限会社)([^\s\n]+)',
            r'([^\s\n]+)(株式会社|合同会社|有限会社)',
        ]
    
    def extract_fields(self, ocr_result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract fields from OCR results.
        
        Args:
            ocr_result: OCR result dictionary (containing at least 'text')
            
        Returns:
            Dictionary of extracted fields
        """
        if "text" not in ocr_result:
            logger.warning("OCR result does not contain 'text' key")
            return {}
        
        text = ocr_result["text"]
        
        # Determine document type (invoice or receipt)
        doc_type = self._determine_document_type(text)
        
        # Extract fields based on document type
        if doc_type == "invoice":
            fields = self._extract_invoice_fields(text)
        elif doc_type == "receipt":
            fields = self._extract_receipt_fields(text)
        else:
            fields = self._extract_general_fields(text)
        
        # Add confidence scores if available
        if "confidence" in ocr_result:
            overall_confidence = ocr_result["confidence"]
            
            # Adjust field confidence based on overall OCR confidence
            for field_name in fields:
                if isinstance(fields[field_name], dict) and "confidence" in fields[field_name]:
                    # Field already has a confidence score
                    continue
                
                # Slightly penalize certain field types that are harder to extract
                penalty = {
                    "total_amount": 0.95,
                    "tax": 0.9,
                    "date": 0.97,
                    "company_name": 0.85,
                }.get(field_name, 1.0)
                
                # Set field confidence
                if isinstance(fields[field_name], dict):
                    fields[field_name]["confidence"] = overall_confidence * penalty
                else:
                    fields[field_name] = {
                        "value": fields[field_name],
                        "confidence": overall_confidence * penalty
                    }
        
        return fields
    
    def _determine_document_type(self, text: str) -> str:
        """
        Determine document type from OCR text.
        
        Args:
            text: OCR extracted text
            
        Returns:
            Document type: 'invoice', 'receipt', or 'unknown'
        """
        # Check for invoice keywords
        invoice_keywords = ["請求書", "納品書", "インボイス", "御請求書"]
        for keyword in invoice_keywords:
            if keyword in text:
                return "invoice"
        
        # Check for receipt keywords
        receipt_keywords = ["領収書", "レシート", "受領書", "お買い上げ明細"]
        for keyword in receipt_keywords:
            if keyword in text:
                return "receipt"
        
        # If text is short and has total amount patterns, likely a receipt
        if len(text) < 500 and any(re.search(pattern, text) for pattern in self.amount_patterns):
            return "receipt"
        
        # Default to unknown
        return "unknown"
    
    def _extract_invoice_fields(self, text: str) -> Dict[str, Any]:
        """
        Extract fields specific to invoices.
        
        Args:
            text: OCR text
            
        Returns:
            Dictionary of extracted fields
        """
        fields = self._extract_general_fields(text)
        
        # Extract invoice-specific fields
        invoice_number = self._extract_pattern(text, self.number_patterns, "invoice_number")
        if invoice_number:
            fields["invoice_number"] = invoice_number
        
        # Extract due date (支払期限)
        due_date_patterns = [
            r'支払[い期]限[:\s]*(.+)',
            r'お支払い期限[:\s]*(.+)',
        ]
        due_date = self._extract_pattern(text, due_date_patterns)
        if due_date:
            fields["due_date"] = due_date
        
        # Extract bank account information if present
        bank_info = self._extract_bank_info(text)
        if bank_info:
            fields.update(bank_info)
        
        return fields
    
    def _extract_receipt_fields(self, text: str) -> Dict[str, Any]:
        """
        Extract fields specific to receipts.
        
        Args:
            text: OCR text
            
        Returns:
            Dictionary of extracted fields
        """
        fields = self._extract_general_fields(text)
        
        # Extract receipt-specific fields
        receipt_number = self._extract_pattern(text, self.number_patterns, "receipt_number")
        if receipt_number:
            fields["receipt_number"] = receipt_number
        
        # Extract items and prices
        items = self._extract_items(text)
        if items:
            fields["items"] = items
        
        return fields
    
    def _extract_general_fields(self, text: str) -> Dict[str, Any]:
        """
        Extract general fields common to invoices and receipts.
        
        Args:
            text: OCR text
            
        Returns:
            Dictionary of extracted fields
        """
        fields = {}
        
        # Extract date
        date = self._extract_pattern(text, self.date_patterns, "date")
        if date:
            fields["date"] = date
        
        # Extract amounts
        amount = self._extract_amount(text)
        if amount:
            fields["total_amount"] = amount
        
        # Extract tax information
        tax = self._extract_tax(text)
        if tax:
            fields["tax"] = tax
        
        # Extract company information
        company_info = self._extract_company_info(text)
        if company_info:
            fields.update(company_info)
        
        return fields
    
    def _extract_pattern(
        self, 
        text: str, 
        patterns: List[Union[str, Pattern]], 
        field_name: Optional[str] = None
    ) -> Optional[str]:
        """
        Extract pattern from text using regex.
        
        Args:
            text: Input text
            patterns: List of regex patterns to try
            field_name: Field name for logging
            
        Returns:
            Extracted text or None
        """
        if not self.use_regex:
            return None
        
        for pattern in patterns:
            match = re.search(pattern, text)
            if match:
                if len(match.groups()) == 1:
                    result = match.group(1).strip()
                else:
                    # Join multiple groups based on pattern
                    if "date" in str(pattern):
                        # For date patterns
                        era = match.group(1) if len(match.groups()) > 3 else ""
                        year = match.group(2) if len(match.groups()) > 1 else ""
                        month = match.group(3) if len(match.groups()) > 2 else ""
                        day = match.group(4) if len(match.groups()) > 3 else match.group(3)
                        
                        # Convert Japanese era to western year if needed
                        if era == "令和":
                            year = str(int(year) + 2018)
                        elif era == "平成":
                            year = str(int(year) + 1988)
                        elif era == "昭和":
                            year = str(int(year) + 1925)
                        
                        result = f"{year}年{month}月{day}日" if year else f"{month}月{day}日"
                    else:
                        # For other patterns, use the most meaningful group
                        result = match.group(len(match.groups())).strip()
                
                logger.debug(f"Extracted {field_name or 'pattern'}: {result}")
                return result
        
        return None
    
    def _extract_amount(self, text: str) -> Optional[str]:
        """
        Extract total amount from text.
        
        Args:
            text: Input text
            
        Returns:
            Extracted amount or None
        """
        # Look for specific total amount patterns first
        total_patterns = [
            r'合計金額[:\s]*([\d,]+)円?',
            r'合計[:\s]*([\d,]+)円?',
            r'総額[:\s]*([\d,]+)円?',
            r'Total[:\s]*¥\s?([\d,]+)',
        ]
        
        # Try to find total amount
        amount = self._extract_pattern(text, total_patterns, "total_amount")
        if amount:
            return amount
        
        # If specific total not found, try to find the largest amount
        all_amounts = []
        for pattern in self.amount_patterns:
            for match in re.finditer(pattern, text):
                if len(match.groups()) >= 1:
                    # Extract the amount value
                    amount_str = match.group(1).replace(',', '')
                    try:
                        amount_val = int(amount_str)
                        all_amounts.append((amount_val, match.group(0)))
                    except ValueError:
                        pass
        
        # If we found amounts, return the largest one
        if all_amounts:
            all_amounts.sort(reverse=True)
            logger.debug(f"Extracted total amount: {all_amounts[0][1]}")
            return all_amounts[0][1]
        
        return None
    
    def _extract_tax(self, text: str) -> Optional[Dict[str, Any]]:
        """
        Extract tax information.
        
        Args:
            text: Input text
            
        Returns:
            Dictionary with tax information or None
        """
        tax_patterns = [
            r'消費税[:\s]*([\d,]+)円?',
            r'税\(?(\d+)%\)?\s*([\d,]+)円?',
            r'消費税\(?(\d+)%\)?\s*([\d,]+)円?',
        ]
        
        for pattern in tax_patterns:
            match = re.search(pattern, text)
            if match:
                if len(match.groups()) == 1:
                    # Just tax amount
                    return {
                        "amount": match.group(1).strip(),
                        "type": "消費税"
                    }
                elif len(match.groups()) >= 2:
                    # Tax rate and amount
                    return {
                        "rate": match.group(1).strip() + "%",
                        "amount": match.group(2).strip(),
                        "type": "消費税"
                    }
        
        return None
    
    def _extract_company_info(self, text: str) -> Dict[str, Any]:
        """
        Extract company information.
        
        Args:
            text: Input text
            
        Returns:
            Dictionary with company information
        """
        company_info = {}
        
        # Extract company name
        for pattern in self.company_patterns:
            match = re.search(pattern, text)
            if match:
                if len(match.groups()) == 2:
                    if match.group(1) in ["株式会社", "合同会社", "有限会社"]:
                        company_name = match.group(1) + match.group(2)
                    else:
                        company_name = match.group(2) + match.group(1)
                    
                    company_info["company_name"] = company_name
                    break
        
        # Extract postal code
        postal_code = self._extract_pattern(text, self.postal_patterns, "postal_code")
        if postal_code:
            company_info["postal_code"] = postal_code
        
        # Extract address (simple heuristic)
        if "postal_code" in company_info:
            address_pattern = company_info["postal_code"] + r'\s*([^\n]{5,50})'
            address = self._extract_pattern(text, [address_pattern], "address")
            if address:
                company_info["address"] = address
        
        # Extract phone
        phone = self._extract_pattern(text, self.phone_patterns, "phone")
        if phone:
            company_info["phone"] = phone
        
        # Extract email
        email = self._extract_pattern(text, self.email_patterns, "email")
        if email:
            company_info["email"] = email
        
        return company_info
    
    def _extract_bank_info(self, text: str) -> Dict[str, Any]:
        """
        Extract bank account information.
        
        Args:
            text: Input text
            
        Returns:
            Dictionary with bank information
        """
        bank_info = {}
        
        # Bank name patterns
        bank_patterns = [
            r'振込先[:\s]*([^\n]{1,20})銀行',
            r'([^\s\n]+)銀行',
        ]
        
        # Branch name patterns
        branch_patterns = [
            r'([^\s\n]+)支店',
        ]
        
        # Account number patterns
        account_patterns = [
            r'口座番号[:\s]*(\d+)',
            r'普通口座[:\s]*(\d+)',
            r'当座預金?[:\s]*(\d+)',
        ]
        
        # Account name patterns
        account_name_patterns = [
            r'名義[:\s]*([^\n]{1,30})',
            r'口座名義[:\s]*([^\n]{1,30})',
        ]
        
        # Extract bank name
        bank_name = self._extract_pattern(text, bank_patterns, "bank_name")
        if bank_name:
            bank_info["bank_name"] = bank_name + "銀行"
        
        # Extract branch name
        branch_name = self._extract_pattern(text, branch_patterns, "branch_name")
        if branch_name:
            bank_info["branch_name"] = branch_name + "支店"
        
        # Extract account number
        account_number = self._extract_pattern(text, account_patterns, "account_number")
        if account_number:
            bank_info["account_number"] = account_number
        
        # Extract account name
        account_name = self._extract_pattern(text, account_name_patterns, "account_name")
        if account_name:
            bank_info["account_name"] = account_name
        
        # Only return if we have at least some bank information
        return bank_info if bank_info else {}
    
    def _extract_items(self, text: str) -> List[Dict[str, str]]:
        """
        Extract itemized list of products/services and prices.
        
        Args:
            text: Input text
            
        Returns:
            List of dictionaries with item information
        """
        items = []
        
        # Split into lines for item extraction
        lines = text.split('\n')
        
        # Pattern for item line with price: "Item name: ¥1,000" or "商品名 1,000円"
        item_pattern = r'([^\d:：]+)[:\s：]+([\d,]+)円?'
        
        for line in lines:
            match = re.search(item_pattern, line)
            if match:
                name = match.group(1).strip()
                price = match.group(2).strip()
                
                # Skip if name is too short or contains keywords indicating it's not an item
                if len(name) < 2 or any(word in name for word in ["合計", "小計", "税", "総額", "Total"]):
                    continue
                
                items.append({
                    "name": name,
                    "price": "¥" + price if not "¥" in price else price
                })
        
        return items
    
    def process_ocr_result(self, ocr_result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process OCR result to extract structured information.
        
        Args:
            ocr_result: OCR result dictionary
            
        Returns:
            Dictionary with OCR result and extracted fields
        """
        # Extract fields
        fields = self.extract_fields(ocr_result)
        
        # Add fields to OCR result
        processed_result = ocr_result.copy()
        processed_result["fields"] = fields
        processed_result["document_type"] = self._determine_document_type(ocr_result.get("text", ""))
        
        return processed_result